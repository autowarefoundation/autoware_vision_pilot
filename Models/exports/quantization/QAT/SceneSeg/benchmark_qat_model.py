#!/usr/bin/env python3
"""
Simple PT2E Converted Model Benchmark Script

A procedural script to benchmark a PyTorch model that was produced by the
`convert_pt2e` function. It loads a QAT checkpoint, converts it, and then
runs validation, calculating mIoU based on successfully processed samples.

Usage:
    python benchmark_converted_pytorch_model.py \\
        --qat_checkpoint_path /path/to/qat_checkpoint.pth
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import cv2
import logging
from tqdm import tqdm
import torch
from typing import List, Dict, Any, Optional
from torchvision import transforms
import onnxruntime as ort
import random
from pathlib import Path

# --- Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(__file__).resolve().parents[5]
models_root = repo_root / "Models"
for path in (repo_root, models_root):
    if str(path) not in sys.path:
        sys.path.append(str(path))


try:
    from data_utils.load_data_scene_seg import LoadDataSceneSeg
    from data_utils.augmentations import Augmentations
except ImportError:
    sys.path.append(str(models_root))
    from data_utils.load_data_scene_seg import LoadDataSceneSeg
    from data_utils.augmentations import Augmentations

from torch.export import export_for_training
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
from torch.ao.quantization import move_exported_model_to_eval

# --- Project Imports ---
sys.path.append(str(models_root))
from data_utils.load_data_scene_seg import LoadDataSceneSeg
from data_utils.augmentations import Augmentations
# IMPORTANT: Use the original FP32 network, not the one with Quantized stubs
from model_components.scene_seg_network import SceneSegNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def create_onnx_from_qat_checkpoint(
    qat_checkpoint_path: str,
    device: torch.device,
    num_calibration_samples: int = 200
) -> Optional[str]:
    """
    Loads a QAT checkpoint, calibrates, converts, and exports to a final INT8 ONNX model.

    Args:
        qat_checkpoint_path: Path to the QAT model checkpoint.
        device: The device to use for model preparation.
        num_calibration_samples: The number of samples to use for calibration.

    Returns:
        The file path to the generated ONNX model, or None if an error occurred.
    """
    try:
        # Step 1: Create the base model structure and load the checkpoint
        logger.info("--- [1/4] Loading QAT checkpoint ---")
        model = SceneSegNetwork().to(device).eval()
        try:
            checkpoint = torch.load(qat_checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded weights from {qat_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return None

        # Step 2: Prepare the model for PT2E QAT
        logger.info("--- [2/4] Preparing model for quantization ---")
        example_input = torch.randn(1, 3, 320, 640).to(device)
        exported_model = export_for_training(model, (example_input,)).module()
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))
        prepared_model = prepare_qat_pt2e(exported_model, quantizer)

        # Allow train/eval mode switching for exported model
        move_exported_model_to_eval(prepared_model)
        # prepared_model.eval()

        # Step 3: Calibrate the model
        logger.info(f"--- [3/4] Calibrating model with {num_calibration_samples} samples ---")
        # Load all validation datasets for calibration
        datasets = {
            "ACDC": LoadDataSceneSeg(os.path.join("<path to SceneSeg>", 'ACDC/ACDC/gt_masks/'),
                                   os.path.join("<path to SceneSeg>", 'ACDC/ACDC/images/'), 'ACDC'),
            "IDDAW": LoadDataSceneSeg(os.path.join("<path to SceneSeg>", 'IDDAW/IDDAW/gt_masks/'),
                                    os.path.join("<path to SceneSeg>", 'IDDAW/IDDAW/images/'), 'IDDAW'),
            "MUSES": LoadDataSceneSeg(os.path.join("<path to SceneSeg>", 'MUSES/MUSES/gt_masks/'),
                                    os.path.join("<path to SceneSeg>", 'MUSES/MUSES/images/'), 'MUSES'),
            "MAPILLARY": LoadDataSceneSeg(os.path.join("<path to SceneSeg>", 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'),
                                        os.path.join("<path to SceneSeg>", 'Mapillary_Vistas/Mapillary_Vistas/images/'), 'MAPILLARY'),
            "COMMA10K": LoadDataSceneSeg(os.path.join("<path to SceneSeg>", 'comma10k/comma10k/gt_masks/'),
                                       os.path.join("<path to SceneSeg>", 'comma10k/comma10k/images/'), 'COMMA10K')
        }
        all_calib_samples = []
        for name, dataset in datasets.items():
            num_val_samples = dataset.getItemCount()[1]
            all_calib_samples.extend([(name, i) for i in range(num_val_samples)])
        random.shuffle(all_calib_samples)
        calibration_set = all_calib_samples[:num_calibration_samples]
        
        image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        with torch.no_grad():
            for dataset_key, sample_idx in tqdm(calibration_set, desc="Calibrating"):
                dataset = datasets[dataset_key]
                img_np, gt_list, _ = dataset.getItemVal(sample_idx)
                augmenter = Augmentations(is_train=False, data_type='SEGMENTATION')
                augmenter.setDataSeg(img_np, gt_list)
                img_aug, _ = augmenter.applyTransformSeg(image=img_np, ground_truth=gt_list)
                calib_tensor = image_transformer(img_aug).unsqueeze(0).to(device)
                prepared_model(calib_tensor)
        logger.info("Calibration complete.")

        # Step 4: Convert to a final INT8 model and export to ONNX
        logger.info("--- [4/4] Converting to INT8 and exporting to ONNX ---")
        # Workaround for device handling during conversion
        torch.set_default_device("cpu")
        prepared_model = prepared_model.to("cpu")
        prepared_model.recompile()
        converted_model = convert_pt2e(prepared_model)
        torch.set_default_device(device.type) # Revert to original device
        
        # Create output directory and final path for the ONNX model
        onnx_output_dir = os.path.join(os.path.dirname(qat_checkpoint_path), "onnx_models")
        os.makedirs(onnx_output_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_output_dir, "SceneSegNetwork_QAT_INT8_final.onnx")
        
        torch.onnx.export(
                converted_model,
                example_input.to("cpu"),
                onnx_path,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamo=True
            )
        
        logger.info(f"Successfully exported final INT8 ONNX model to: {onnx_path}")
        return onnx_path

    except Exception as e:
        logger.error(f"An error occurred during the ONNX creation process: {e}", exc_info=True)
        return None

def calculate_iou(output: np.ndarray, label: np.ndarray) -> float:
    """Calculate IoU between prediction and ground truth masks."""
    intersection = np.logical_and(label, output)
    union = np.logical_or(label, output)
    return (np.sum(intersection) + 1) / float(np.sum(union) + 1)

def run_onnx_inference(
    onnx_path: str,
    input_tensor: torch.Tensor,
    device: str = "cpu"
) -> np.ndarray:
    """Run inference using ONNX Runtime."""
    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_data = input_tensor.cpu().numpy()
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]

def main():
    """Main function to run the benchmarking process."""
    parser = argparse.ArgumentParser(
        description="""
        QAT Checkpoint to ONNX Conversion & Validation Script.
        This script takes a trained QAT checkpoint, performs calibration,
        converts it to a final INT8 ONNX model, and runs validation on it.
        """
    )
    parser.add_argument("--qat_checkpoint_path", type=str, required=True,
                        help="Path to the QAT checkpoint to convert and benchmark.")
    parser.add_argument("--dataset_root", type=str,
                        default="<path to SceneSeg>",
                        help="Root directory of datasets.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model preparation ('cpu' or 'cuda'). Inference is on CPU.")
    parser.add_argument("--num_samples_per_dataset", type=int, default=20,
                        help="Number of validation samples to process from each dataset.")
    parser.add_argument("--num_calibration_samples", type=int, default=200,
                        help="Number of samples to use for calibration.")
    args = parser.parse_args()
    device = torch.device(args.device)

    # --- 1. Create Final ONNX Model from QAT Checkpoint ---
    onnx_model_path = create_onnx_from_qat_checkpoint(
        qat_checkpoint_path=args.qat_checkpoint_path,
        device=device,
        num_calibration_samples=args.num_calibration_samples
    )
    if not onnx_model_path:
        logger.error("Failed to create ONNX model. Exiting.")
        return 1

    # --- 2. Load All Datasets for Validation ---
    logger.info("--- Loading Datasets for Validation ---")
    
    # Define paths
    root = args.dataset_root
    datasets_to_load = {
        'ACDC': (os.path.join(root, 'ACDC/ACDC/gt_masks/'), os.path.join(root, 'ACDC/ACDC/images/')),
        'IDDAW': (os.path.join(root, 'IDDAW/IDDAW/gt_masks/'), os.path.join(root, 'IDDAW/IDDAW/images/')),
        'MUSES': (os.path.join(root, 'MUSES/MUSES/gt_masks/'), os.path.join(root, 'MUSES/MUSES/images/')),
        'MAPILLARY': (os.path.join(root, 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'), os.path.join(root, 'Mapillary_Vistas/Mapillary_Vistas/images/')),
        'COMMA10K': (os.path.join(root, 'comma10k/comma10k/gt_masks/'), os.path.join(root, 'comma10k/comma10k/images/')),
        'BDD100K': (os.path.join(root, 'BDD100K/BDD100K/gt_masks/'), os.path.join(root, 'BDD100K/BDD100K/images/'))
    }
    
    loaded_datasets = {}
    total_val_samples = 0
    for name, (labels_fp, images_fp) in datasets_to_load.items():
        dataset = LoadDataSceneSeg(labels_fp, images_fp, name)
        _, num_val = dataset.getItemCount()
        logger.info(f"{name}: {num_val} validation samples found.")
        loaded_datasets[name] = dataset
        total_val_samples += num_val
    logger.info(f"{total_val_samples} total validation samples found.")

    # --- 3. Benchmark Final ONNX Model ---
    logger.info(f"--- Benchmarking Final ONNX Model: {onnx_model_path} ---")
    running_iou_full, running_iou_bg, running_iou_fg, running_iou_rd = 0.0, 0.0, 0.0, 0.0
    total_processed_samples = 0
    total_inference_time = 0.0

    # Define the image transformer once
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for dataset_name, dataset in loaded_datasets.items():
        _, num_val = dataset.getItemCount()
        samples_to_process = min(num_val, args.num_samples_per_dataset)
        logger.info(f"Benchmarking {dataset_name} ({samples_to_process} samples)...")
        
        for i in tqdm(range(samples_to_process), desc=f"Processing {dataset_name}"):
            img_np, gt_list, _ = dataset.getItemVal(i)
            
            # Process image
            augmenter = Augmentations(is_train=False, data_type='SEGMENTATION')
            augmenter.setDataSeg(img_np, gt_list)
            img_aug, _ = augmenter.applyTransformSeg(image=img_np, ground_truth=gt_list)
            img_tensor = image_transformer(img_aug).unsqueeze(0) # Keep on CPU for ONNX

            # Run ONNX inference
            start_inference = time.perf_counter()
            output_raw = run_onnx_inference(onnx_model_path, img_tensor)
            inference_time = (time.perf_counter() - start_inference) * 1000

            # Process output
            output_val = np.transpose(output_raw.squeeze(0), (1, 2, 0))
            output_processed = np.zeros_like(output_val)
            pred_indices = np.argmax(output_val, axis=2)
            output_processed[np.arange(pred_indices.shape[0])[:, None],
                           np.arange(pred_indices.shape[1]), pred_indices] = 1

            # Process ground truth
            gt_fused = np.stack((gt_list[1], gt_list[2], gt_list[3]), axis=2)
            h, w = output_processed.shape[:2]
            gt_fused_resized = cv2.resize(gt_fused.astype(np.uint8), (w, h),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)

            # Calculate metrics
            running_iou_full += calculate_iou(output_processed, gt_fused_resized)
            running_iou_bg += calculate_iou(output_processed[:,:,0], gt_fused_resized[:,:,0])
            running_iou_fg += calculate_iou(output_processed[:,:,1], gt_fused_resized[:,:,1])
            running_iou_rd += calculate_iou(output_processed[:,:,2], gt_fused_resized[:,:,2])
            total_inference_time += inference_time
            total_processed_samples += 1

    # --- 4. Report Final Results ---
    logger.info("\n=== Final ONNX Benchmark Results ===")
    if total_processed_samples > 0:
        metrics = {
            "mIoU_full": running_iou_full / total_processed_samples,
            "mIoU_bg": running_iou_bg / total_processed_samples,
            "mIoU_fg": running_iou_fg / total_processed_samples,
            "mIoU_rd": running_iou_rd / total_processed_samples,
            "avg_inference_time": total_inference_time / total_processed_samples,
            "total_samples": total_processed_samples
        }
        logger.info(f"mIoU (full): {metrics['mIoU_full']:.4f}")
        logger.info(f"Class mIoUs: BG={metrics['mIoU_bg']:.4f}, FG={metrics['mIoU_fg']:.4f}, Road={metrics['mIoU_rd']:.4f}")
        logger.info(f"Avg. inference time: {metrics['avg_inference_time']:.2f} ms")
        logger.info(f"Total samples processed: {metrics['total_samples']}")
    else:
        logger.warning("No samples were processed during validation.")

    return 0


if __name__ == "__main__":
    sys.exit(main()) 
