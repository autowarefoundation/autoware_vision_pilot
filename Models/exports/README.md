# Quanty - ONNX Model Quantization and Benchmarking Tools

![Quanty GIF](../../Media/Quanty_GIF.gif) ![Quanty GIF 2](../../Media/Quanty_GIF_2.gif)

This folder provides comprehensive tools for ONNX model optimization and performance analysis using Quantization-Aware Training (QAT):

1. `quantization/QAT/SceneSeg/quantize_model_sceneseg.py`: Performs QAT training and converts FP32 models to optimized INT8 versions
2. `quantization/QAT/SceneSeg/benchmark_qat_model.py`: Converts QAT checkpoints to final ONNX models and benchmarks their performance
3. `quantization/QAT/AutoDrive/autodrive_qat.py`: AutoDrive QAT ONNX export and benchmark helper
4. `quantization/PTQ/AutoDrive/autodrive_ptq.py`: AutoDrive PT2E PTQ export and benchmark helper
5. `benchmark_onnx_models.py`: Benchmarks and compares PyTorch, ONNX FP32, and ONNX INT8 model performance
6. Additional export utilities can be found in the **libtorch** and **onnx_rt** folders.

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/1o7lEQHD0VcooGQ0-rRwHe5UGDCBkUCNB/view?usp=drive_link)

## Features

### Quantization-Aware Training (QAT) Tool
- Fine-tunes pre-trained FP32 models with quantization-aware training for optimal INT8 performance
- Uses PyTorch 2.0 Export (PT2E) framework with XNNPACKQuantizer
- Employs advanced QAT recipes (observer freezing, batch norm freezing)
- Supports symmetric quantization configuration for maximum compatibility
- Includes comprehensive validation during training with mIoU metrics
- Produces highly optimized INT8 models with minimal accuracy loss

### QAT Benchmark Tool
- Takes QAT checkpoints and converts them to final INT8 ONNX models
- Performs calibration using validation datasets for optimal quantization parameters
- Benchmarks the final ONNX model with comprehensive mIoU metrics
- Measures inference speed and accuracy on real validation data
- Provides complete end-to-end validation of the QAT workflow

### Multi-Model Benchmarking Tool
- Measures inference speed, model size, and memory usage
- Evaluates output accuracy using mean Intersection over Union (mIoU)
- Provides class-specific IoU metrics (Background, Foreground, Road)
- Uses model-centric evaluation approach (each model processes all samples)
- Ensures IoU calculation consistency with SceneSegTrainer methodology
- Supports visualization of model outputs with side-by-side comparisons
- Uses real validation data for accurate benchmarking
- Provides detailed performance metrics and statistics per model and dataset

## Quantization-Aware Training Workflow

The QAT process fine-tunes a pre-trained FP32 model to learn weights that are resilient to quantization, significantly reducing accuracy loss compared to post-training quantization methods.

### QAT Process Overview:
1. **Load pre-trained FP32 model**: Starting point with good baseline accuracy
2. **PT2E Export**: Export model graph for training using PyTorch 2.0 framework
3. **Quantization Preparation**: Insert fake quantization nodes using XNNPACKQuantizer
4. **Fine-tuning**: Train with quantization-aware loss for several epochs
5. **Advanced Recipes**: Freeze observers and batch norm statistics for stability
6. **Validation**: Continuous monitoring with mIoU metrics across all datasets

## Usage

### 1. Quantization-Aware Training

Fine-tune your FP32 model with quantization awareness:

```bash
python Models/exports/quantization/QAT/SceneSeg/quantize_model_sceneseg.py \
    --model_save_root_path /path/to/save/checkpoints/ \
    --root /path/to/dataset/root/ \
    --fp32_model /path/to/pretrained_fp32_model.pth \
    --epochs 5
```

**Key Arguments:**
- `--model_save_root_path`: Directory where QAT checkpoints will be saved
- `--root`: Root directory containing all datasets (ACDC, IDDAW, MUSES, etc.)
- `--fp32_model`: Path to the starting FP32 model weights
- `--epochs`: Number of epochs to fine-tune the model (typically 3-5)

**Output:**
- QAT checkpoints saved periodically during training
- Final calibrated model: `qat_model_final_calibrated.pth`
- Comprehensive validation scores throughout training

### 2. QAT Model Conversion and Benchmarking

Convert QAT checkpoint to final ONNX model and benchmark performance:

```bash
python Models/exports/quantization/QAT/SceneSeg/benchmark_qat_model.py \
    --qat_checkpoint_path /path/to/qat_checkpoint_epoch_4_step_31057.pth \
    --dataset_root /path/to/dataset/root/ \
    --num_calibration_samples 200 \
    --num_samples_per_dataset 20 \
    --device cuda
```

**Key Arguments:**
- `--qat_checkpoint_path`: Path to the QAT checkpoint from step 1
- `--dataset_root`: Root directory for datasets (used for calibration and validation)
- `--num_calibration_samples`: Number of samples for final calibration (default: 200)
- `--num_samples_per_dataset`: Number of samples for validation per dataset (default: 20)
- `--device`: Device for model preparation ('cpu' or 'cuda')

**What this script does:**
1. **Load QAT checkpoint**: Reconstructs the quantization-aware model
2. **Calibrate**: Uses validation data to determine optimal quantization parameters
3. **Convert to INT8**: Creates final optimized INT8 model using PT2E convert
4. **Export ONNX**: Saves the model as `SceneSegNetwork_QAT_INT8_final.onnx`
5. **Benchmark**: Runs comprehensive validation with mIoU metrics and timing

**Output:**
- Final ONNX model: `SceneSegNetwork_QAT_INT8_final.onnx` in `onnx_models/` subdirectory
- Detailed validation scores: Overall and per-class mIoU
- Average inference time measurements

### 3. Multi-Model Benchmarking

Compare PyTorch, FP32, and INT8 model performance:

```bash
python benchmark_onnx_models.py \
    --pytorch_model_path /path/to/pytorch_model.pth \
    --fp32_model_path /path/to/fp32_model.onnx \
    --int8_model_path /path/to/int8_model.onnx \
    --dataset_root /path/to/datasets/ \
    --num_samples_per_dataset 20 \
    --visualizations_per_dataset 2 \
    --save_outputs \
    --device cuda
```

The benchmark tool will:
1. Load all specified models
2. Process samples from each available dataset 
3. Calculate IoU scores (overall and per-class)
4. Measure inference times
5. Generate combined visualizations (Original | GT | Prediction)
6. Produce a comprehensive comparison report

#### Optional Parameters:
- `--pytorch_target_hw` - Target height,width for PyTorch model (e.g., "320,640")
- `--onnx_target_hw` - Target height,width for ONNX models (derived from model if not specified)
- `--warmup_runs` - Number of warmup inference runs (default: 3)
- `--output_dir_base` - Directory for saving results (default: "./benchmark_multimodel_results")

## QAT Training Parameters

The QAT process uses carefully tuned parameters for optimal results:

- **Learning Rate**: 1e-5 (conservative for fine-tuning)
- **Batch Size**: 32 (balanced for memory and convergence)
- **Observer Freezing**: After 2 epochs (prevents overfitting to quantization)
- **Batch Norm Freezing**: After 1 epoch (maintains stable statistics)
- **Quantization Config**: Symmetric, per-tensor (maximum compatibility)

## Calibration Dataset Structure

The QAT process expects datasets organized in the following structure:

```
SceneSeg/
├── ACDC/
│   └── ACDC/
│       ├── images/
│       │   ├── image1.jpg
│       │   └── ...
│       └── gt_masks/
│           ├── mask1.png
│           └── ...
├── IDDAW/
│   └── IDDAW/
│       ├── images/
│       │   ├── image1.jpg
│       │   └── ...
│       └── gt_masks/
│           ├── mask1.png
│           └── ...
├── MUSES/
│   └── MUSES/
│       ├── images/
│       └── gt_masks/
├── Mapillary_Vistas/
│   └── Mapillary_Vistas/
│       ├── images/
│       └── gt_masks/
└── comma10k/
    └── comma10k/
        ├── images/
        └── gt_masks/
```

Each dataset should contain RGB images and corresponding ground truth masks. The QAT process will automatically use all available datasets for training and validation.

## Benefits of QAT over Post-Training Quantization

- **Superior accuracy preservation**: QAT typically maintains 95-98% of original accuracy vs 85-90% for PTQ
- **Hardware optimization**: Models are optimized during training for target quantization scheme
- **Robust quantization**: Weights learn to be resilient to quantization noise
- **Reduced calibration sensitivity**: Less dependent on calibration dataset quality
- **Better convergence**: Gradual adaptation to quantization during training

## Final Directory Structure

After QAT training, your output directory will contain:

```
/path/to/save/checkpoints/
├── qat_checkpoint_epoch_0_step_7999.pth
├── qat_checkpoint_epoch_1_step_15999.pth
├── qat_checkpoint_epoch_2_step_23999.pth
├── qat_checkpoint_epoch_3_step_31999.pth
├── qat_checkpoint_epoch_4_step_31057.pth
├── qat_model_final_calibrated.pth
└── onnx_models/
    └── SceneSegNetwork_QAT_INT8_final.onnx
```

The final ONNX model can be used directly for deployment in production environments or with inference frameworks like ONNX Runtime.

## Dependencies
- torch (with quantization support)
- onnx
- onnxruntime
- numpy
- opencv-python
- PIL (Pillow)
- tqdm (optional, for progress bars)
- torchvision

## Testing
The QAT tools have been tested with:
- Various scene segmentation model architectures
- Different input sizes and resolutions (320x640 default)
- Multiple validation datasets (ACDC, IDDAW, MUSES, Mapillary, comma10k)
- Different hardware configurations (CPU and CUDA)
- PyTorch 2.0+ with PT2E export framework

## Additional Utilities

### trace_pytorch_model.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks via a .pth checkpoint file, trace and then export the traced model as a *.pt file.

#### Example Usage
```bash
python3 trace_pytorch_model.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/traced_model.pt
```

### convert_pytorch_to_onnx.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks and convert and export that model to ONNX format at FP32 precision as a *.onnx file.

#### Example Usage
```bash
python3 convert_pytorch_to_onnx.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/converted_model.onnx
```
