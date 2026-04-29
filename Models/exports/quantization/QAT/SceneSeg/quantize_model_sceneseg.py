#!/usr/bin/env python3

import os
import torch
import random
from argparse import ArgumentParser
import sys
import copy
import logging
import numpy as np
from torchvision import transforms
import cv2 # Import OpenCV for resizing
from dataclasses import dataclass
from typing import Optional
import itertools
import torch.nn.functional as F
from pathlib import Path

# --- PT2E and ONNX Imports ---
import torch.quantization
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.export import export_for_training
import torch.onnx
import torch.nn as nn
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

# Make sure we can find required modules after moving under exports/quantization/QAT/SceneSeg.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODELS_ROOT = _REPO_ROOT / "Models"
for _path in (_REPO_ROOT, _MODELS_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

# Import required modules
from data_utils.load_data_scene_seg import LoadDataSceneSeg
from data_utils.augmentations import Augmentations
from model_components.scene_seg_network import SceneSegNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions for Validation ---
def calculate_iou(output, label):
    """Helper function for IoU calculation."""
    intersection = np.logical_and(label, output)
    union = np.logical_or(label, output)
    iou_score = (np.sum(intersection) + 1) / float(np.sum(union) + 1)
    return iou_score

def run_validation(model, dataset_loader, num_samples, device):
    """Runs validation on the model and returns mIoU scores."""
    logger.info("Running validation...")
    
    # Ensure model is on the correct device and in eval mode
    model.to(device)
    torch.ao.quantization.move_exported_model_to_eval(model)

    running_iou_full, running_iou_bg, running_iou_fg, running_iou_rd = 0.0, 0.0, 0.0, 0.0

    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(num_samples):
            img_np, gt_list, _ = dataset_loader.getItemVal(i)
            
            # CRITICAL FIX: Add the missing augmentation/resizing step for validation images
            augmenter = Augmentations(is_train=False, data_type='SEGMENTATION')
            augmenter.setDataSeg(img_np, gt_list)
            # We only need the augmented image for input, gt is handled later
            img_aug, _ = augmenter.applyTransformSeg(image=img_np, ground_truth=gt_list)

            # Preprocess image and ground truth
            # Use the resized image (img_aug) instead of the raw one (img_np)
            image_tensor = image_transformer(img_aug).unsqueeze(0).to(device)
            gt_fused = np.stack((gt_list[1], gt_list[2], gt_list[3]), axis=2)

            # Get model prediction
            prediction = model(image_tensor)
            
            # Post-process prediction to a one-hot numpy array
            output_val = prediction.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
            output_processed = np.zeros_like(output_val)
            pred_indices = np.argmax(output_val, axis=2)
            output_processed[np.arange(pred_indices.shape[0])[:, None], np.arange(pred_indices.shape[1]), pred_indices] = 1
            
            # Resize GT to match output shape for fair comparison
            h, w = output_processed.shape[:2]
            gt_fused_resized = cv2.resize(gt_fused.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Calculate IoU scores
            running_iou_full += calculate_iou(output_processed, gt_fused_resized)
            running_iou_bg += calculate_iou(output_processed[:,:,0], gt_fused_resized[:,:,0])
            running_iou_fg += calculate_iou(output_processed[:,:,1], gt_fused_resized[:,:,1])
            running_iou_rd += calculate_iou(output_processed[:,:,2], gt_fused_resized[:,:,2])

    mIoU_full = running_iou_full / num_samples
    mIoU_bg = running_iou_bg / num_samples
    mIoU_fg = running_iou_fg / num_samples
    mIoU_rd = running_iou_rd / num_samples
    
    logger.info(f"Validation mIoU: Full={mIoU_full:.4f}, BG={mIoU_bg:.4f}, FG={mIoU_fg:.4f}, Road={mIoU_rd:.4f}")
    return mIoU_full, mIoU_bg, mIoU_fg, mIoU_rd

def main_qat():
    parser = ArgumentParser(description="PT2E QAT for Scene Segmentation")
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", required=True, 
                        help="Root path to save model checkpoints and final ONNX model")
    parser.add_argument("-r", "--root", dest="root", required=True, 
                        help="Root path to folder where training data is stored")
    parser.add_argument("-fp32", "--fp32_model", dest="fp32_model_path", 
                        default="best_model.pth",
                        help="Path to pretrained FP32 model")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=5,
                        help="Number of QAT epochs")

    args = parser.parse_args()

    qat_params = {
        "learning_rate": 1e-5,  # Reverted to a stable default
        "batch_size": 32,       # Reverted to a stable default
    }
    # --- Advanced QAT Recipe Parameters ---
    num_observer_update_epochs = 2  # Freeze observers after this many epochs
    num_batch_norm_update_epochs = 1 # Freeze BN stats after this many epochs

    # Root path
    root = args.root
    model_save_root_path = args.model_save_root_path

    # Data paths
    # ACDC
    acdc_labels_filepath = root + 'ACDC/ACDC/gt_masks/'
    acdc_images_filepath = root + 'ACDC/ACDC/images/'

    # IDDAW
    iddaw_labels_fileapath = root + 'IDDAW/IDDAW/gt_masks/'
    iddaw_images_fileapath = root + 'IDDAW/IDDAW/images/'

    # MUSES
    muses_labels_fileapath = root + 'MUSES/MUSES/gt_masks/'
    muses_images_fileapath = root + 'MUSES/MUSES/images/'

    # MAPILLARY
    mapillary_labels_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'
    mapillary_images_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/images/'

    # COMMA10K
    comma10k_labels_fileapath = root + 'comma10k/comma10k/gt_masks/'
    comma10k_images_fileapath = root + 'comma10k/comma10k/images/'

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load and Prepare FP32 Model ---
    logger.info("Initializing FP32 model...")
    fp32_model = SceneSegNetwork().to(device)

    # Load Pretrained FP32 Weights
    fp32_path = args.fp32_model_path
    logger.info(f"Loading pretrained weights from: {fp32_path}")
    try:
        fp32_checkpoint = torch.load(fp32_path, map_location=device)
        state_dict = fp32_checkpoint.get("model_state_dict") or fp32_checkpoint.get("state_dict") or fp32_checkpoint
        fp32_model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded pretrained weights.")
    except Exception as e:
        logger.error(f"Error loading FP32 checkpoint: {e}. Exiting.")
        return

    # --- 2. Prepare Model for PT2E QAT ---
    logger.info("Preparing model for PT2E QAT...")
    example_input = torch.randn(1, 3, 320, 640).to(device)
    
    # Use the modern torch.export API to capture the graph for training
    logger.info("Exporting model graph for training using torch.export...")
    exported_model = export_for_training(fp32_model, (example_input,)).module()
    
    # --- Define Quantization Configuration using the XNNPACK Quantizer ---
    logger.info("Configuring model quantization with XNNPACKQuantizer...")
    
    # First, try with XNNPACKQuantizer with explicit is_qat=True
    quantizer = XNNPACKQuantizer()
    # CRITICAL FIX: We need to explicitly set is_qat=True and align with stable config
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True, is_per_channel=False))
    
    # This inserts fake_quant nodes into the graph using the XNNPACK rules.
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    # print(prepared_model.graph)
    logger.info("Model prepared for PT2E QAT with XNNPACKQuantizer.")

    # --- 3. Create Optimizer for the QAT Model ---
    # CRITICAL FIX: Optimizer is created *after* preparing the model
    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=qat_params["learning_rate"],
        weight_decay=1e-5
    )
    # The loss function will be defined in the loop as it may use class weights
    loss_fn = torch.nn.CrossEntropyLoss()

    # ACDC - Data Loading
    acdc_Dataset = LoadDataSceneSeg(acdc_labels_filepath, acdc_images_filepath, 'ACDC')
    acdc_num_train_samples, acdc_num_val_samples = acdc_Dataset.getItemCount()

    # IDDAW - Data Loading
    iddaw_Dataset = LoadDataSceneSeg(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
    iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadDataSceneSeg(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
    muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

    # Mapillary - Data Loading
    mapillary_Dataset = LoadDataSceneSeg(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

    # comma10k - Data Loading
    comma10k_Dataset = LoadDataSceneSeg(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = acdc_num_train_samples + \
    + iddaw_num_train_samples + muses_num_train_samples \
    + mapillary_num_train_samples + comma10k_num_train_samples
    logger.info(f"{total_train_samples}: total training samples")

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    logger.info(f"{total_val_samples}: total validation samples")

    # --- 4. Run QAT Training Loop ---
    logger.info("Starting QAT training loop...")
    
    # --- QAT Recipe Flags ---
    num_observer_update_flag = True
    num_batch_norm_update_flag = True

    # Training loop
    for epoch in range(0, args.epochs):
        logger.info(f"QAT Epoch {epoch+1}/{args.epochs}")
        
        # Set model to training mode
        torch.ao.quantization.move_exported_model_to_train(prepared_model)

        # --- QAT Recipe: Freeze Observers and Batch Norm stats ---
        if epoch >= num_observer_update_epochs and num_observer_update_flag:
            logger.info(f"Disabling observers for subsequent epochs (epoch {epoch}).")
            prepared_model.apply(torch.ao.quantization.disable_observer)
            num_observer_update_flag = False

        if epoch >= num_batch_norm_update_epochs and num_batch_norm_update_flag:
            logger.info(f"Freezing Batch Norm stats for subsequent epochs (epoch {epoch}).")
            for n in prepared_model.graph.nodes:
                if n.target in [
                    torch.ops.aten._native_batch_norm_legit.default,
                    torch.ops.aten.cudnn_batch_norm.default,
                ]:
                    if len(n.args) > 5:
                        new_args = list(n.args)
                        new_args[5] = False  # The 'training' flag for BN
                        n.args = tuple(new_args)
            prepared_model.recompile()
            num_batch_norm_update_flag = False

        # Create a combined list of all training samples for easier iteration
        all_train_samples = []
        for i in range(acdc_num_train_samples):
            all_train_samples.append(("ACDC", i))
        for i in range(iddaw_num_train_samples):
            all_train_samples.append(("IDDAW", i))
        for i in range(muses_num_train_samples):
            all_train_samples.append(("MUSES", i))
        for i in range(mapillary_num_train_samples):
            all_train_samples.append(("MAPILLARY", i))
        for i in range(comma10k_num_train_samples):
            all_train_samples.append(("COMMA10K", i))
        random.shuffle(all_train_samples)

        # --- Manual Training Step ---
        for i, (dataset_key, sample_idx) in enumerate(all_train_samples):
            # 1. Get Data
            if dataset_key == "ACDC":
                dataset = acdc_Dataset
            elif dataset_key == "IDDAW":
                dataset = iddaw_Dataset
            elif dataset_key == "MUSES":
                dataset = muses_Dataset
            elif dataset_key == "MAPILLARY":
                dataset = mapillary_Dataset
            elif dataset_key == "COMMA10K":
                dataset = comma10k_Dataset
            
            img_np, gt_list, class_weights = dataset.getItemTrain(sample_idx)

            # 2. Augment Data
            augmenter = Augmentations(is_train=False, data_type='SEGMENTATION')
            augmenter.setDataSeg(img_np, gt_list)
            img_aug, augmented_gt = augmenter.applyTransformSeg(image=img_np, ground_truth=gt_list)
            
            # 3. Preprocess and Load Tensors
            image_loader = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            gt_loader = transforms.Compose([transforms.ToTensor()])
            
            image_tensor = image_loader(img_aug).unsqueeze(0).to(device)
            gt_fused = np.stack((augmented_gt[1], augmented_gt[2], augmented_gt[3]), axis=2)
            gt_tensor_one_hot = gt_loader(gt_fused).unsqueeze(0).to(device)
            
            # CRITICAL FIX: Convert one-hot target to class indices for CrossEntropyLoss
            gt_tensor = torch.argmax(gt_tensor_one_hot, dim=1)
            
            # 4. Forward Pass
            optimizer.zero_grad()
            prediction = prepared_model(image_tensor)
            
            # 5. Calculate Loss
            # Pass class indices (gt_tensor) not one-hot (gt_tensor_one_hot)
            loss = loss_fn(prediction, gt_tensor)
            
            # 6. Backward Pass and Optimize
            loss.backward()
            optimizer.step()
            
            # Simplified logging
            if((i+1) % 250 == 0):
                logger.info(f"Loss at step {i+1}: {loss.item():.4f}")
            

            
            # Checkpoint saving during the loop - SIMPLIFIED
            if((i+1) % 8000  == 0 or (i+1) == total_train_samples):
                # We only save the QAT state_dict. The conversion to INT8 is a separate step
                # handled by the dedicated conversion script. This avoids corrupting the training state.
                qat_save_path = os.path.join(model_save_root_path, f'qat_checkpoint_epoch_{epoch}_step_{i}.pth')
                torch.save(prepared_model.state_dict(), qat_save_path)
                logger.info(f"Saved QAT training checkpoint to {qat_save_path}")
                
                # --- Validation (Aligned with train_scene_seg.py) ---
                logger.info('Validating on the entire validation set...')
                torch.ao.quantization.move_exported_model_to_eval(prepared_model)

                running_iou_full, running_iou_bg, running_iou_fg, running_iou_rd = 0.0, 0.0, 0.0, 0.0
                
                # We will not use the run_validation helper here to exactly match
                # the validation logic from the original FP32 training script.
                image_transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                with torch.no_grad():
                    all_val_datasets = {
                        "ACDC": (acdc_Dataset, acdc_num_val_samples),
                        "IDDAW": (iddaw_Dataset, iddaw_num_val_samples),
                        "MUSES": (muses_Dataset, muses_num_val_samples),
                        "MAPILLARY": (mapillary_Dataset, mapillary_num_val_samples),
                        "COMMA10K": (comma10k_Dataset, comma10k_num_val_samples),
                    }

                    for name, (dataset, num_samples) in all_val_datasets.items():
                        logger.info(f"Validating on {name}...")
                        for val_idx in range(num_samples):
                            img_np, gt_list, _ = dataset.getItemVal(val_idx)
                            
                            augmenter = Augmentations(is_train=False, data_type='SEGMENTATION')
                            augmenter.setDataSeg(img_np, gt_list)
                            img_aug, _ = augmenter.applyTransformSeg(image=img_np, ground_truth=gt_list)
                            
                            image_tensor = image_transformer(img_aug).unsqueeze(0).to(device)
                            gt_fused = np.stack((gt_list[1], gt_list[2], gt_list[3]), axis=2)
                            
                            prediction = prepared_model(image_tensor)
                            
                            output_val = prediction.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
                            output_processed = np.zeros_like(output_val)
                            pred_indices = np.argmax(output_val, axis=2)
                            output_processed[np.arange(pred_indices.shape[0])[:, None], np.arange(pred_indices.shape[1]), pred_indices] = 1
                            
                            h, w = output_processed.shape[:2]
                            gt_fused_resized = cv2.resize(gt_fused.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                            running_iou_full += calculate_iou(output_processed, gt_fused_resized)
                            running_iou_bg += calculate_iou(output_processed[:,:,0], gt_fused_resized[:,:,0])
                            running_iou_fg += calculate_iou(output_processed[:,:,1], gt_fused_resized[:,:,1])
                            running_iou_rd += calculate_iou(output_processed[:,:,2], gt_fused_resized[:,:,2])
                
                # Calculate final mIoU across all datasets
                mIoU_full = running_iou_full / total_val_samples
                mIoU_bg = running_iou_bg / total_val_samples
                mIoU_fg = running_iou_fg / total_val_samples
                mIoU_rd = running_iou_rd / total_val_samples
                
                logger.info(f"Overall Validation mIoU: Full={mIoU_full:.4f}, BG={mIoU_bg:.4f}, FG={mIoU_fg:.4f}, Road={mIoU_rd:.4f}")

                # Reset model back to training mode for the next iteration
                torch.ao.quantization.move_exported_model_to_train(prepared_model)

    # --- 5. Final QAT Model Saving ---
    # The primary goal of this script is to produce a well-calibrated QAT model.
    logger.info("QAT training finished. Saving final calibrated QAT model.")
    
    # Ensure model is in eval mode before saving the final version
    torch.ao.quantization.move_exported_model_to_eval(prepared_model)
    
    # Save final QAT model state_dict. This is the artifact that will be
    # passed to the conversion script.
    final_model_path = os.path.join(model_save_root_path, "qat_model_final_calibrated.pth")
    torch.save(prepared_model.state_dict(), final_model_path)
    logger.info(f"Saved final calibrated QAT model state_dict to {final_model_path}")
    logger.info("This file can now be used with `convert_qat_to_fp32_pytorch.py` to create the final INT8 model.")
    
    logger.info("PT2E QAT training workflow complete.")

if __name__ == '__main__':
    main_qat()
