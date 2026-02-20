# AutoSpeed TensorRT Backend

This is a specialized TensorRT inference backend for the **AutoSpeed** object detection model.

## Architecture Overview

```
Input Image (ROS Topic)
    ↓
RunAutoSpeedNode
    ↓
AutoSpeedTensorRTEngine::doInference()
    ├─ preprocessAutoSpeed() [Letterbox + Normalize]
    ├─ TensorRT Inference
    └─ Return Raw Tensor
    ↓
Post-Processing (in RunAutoSpeedNode)
    ├─ Parse Predictions
    ├─ Confidence Filtering
    ├─ NMS (Non-Maximum Suppression)
    └─ Coordinate Transformation
    ↓
Output Detections (vision_msgs::Detection2DArray)
```

## Key Features

### 1. **Flexible Model Input**
- Accepts **PyTorch checkpoints** (`.pt`) or **ONNX models** (`.onnx`)
- Automatically converts PyTorch → ONNX → TensorRT on first run
- Caches TensorRT engines for faster subsequent loads

### 2. **AutoSpeed-Specific Preprocessing**
Unlike the standard TensorRT backend (which uses ImageNet normalization), AutoSpeed uses:
- **Letterbox resizing**: Maintains aspect ratio with gray padding (114, 114, 114)
- **Simple normalization**: Divides by 255.0 (no mean/std normalization)
- **RGB format**: Converts from OpenCV BGR to RGB
- **Stores transformation parameters**: For accurate coordinate mapping back to original image

### 3. **Precision Support**
- **FP32**: Full precision (slower, more accurate)
- **FP16**: Half precision (faster, slightly less accurate, requires GPU support)

### 4. **Post-Processing Pipeline**
- **Confidence filtering**: Removes low-confidence detections
- **NMS**: Eliminates duplicate/overlapping boxes
- **Coordinate transformation**: Maps detections from 640x640 letterbox space back to original image coordinates

## File Structure

```
VisionPilot/
├── common/
│   └── backends/
│       └── autospeed/
│           ├── tensorrt_engine.hpp    # Backend header
│           ├── tensorrt_engine.cpp    # Backend implementation
│           └── README.md              # This file
└── ROS2/
    └── models/
        ├── include/
        │   └── run_autospeed_node.hpp  # ROS2 node header
        └── src/
            └── run_autospeed_node.cpp  # ROS2 node implementation
```

## Usage

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | - | Path to `.pt` or `.onnx` model |
| `precision` | string | "fp32" | "fp32" or "fp16" |
| `gpu_id` | int | 0 | GPU device ID |
| `conf_threshold` | double | 0.6 | Minimum confidence for detections |
| `iou_threshold` | double | 0.45 | IoU threshold for NMS |
| `input_topic` | string | "/sensors/camera/image_raw" | Input image topic |
| `output_topic` | string | "/autospeed/detections" | Output detections topic |

### Example Launch

```python
# Launch file example
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='models',
            executable='autospeed_node_exe',
            name='autospeed',
            parameters=[{
                'model_path': '/path/to/autospeed/best.pt',
                'precision': 'fp16',
                'conf_threshold': 0.6,
                'iou_threshold': 0.45,
                'input_topic': '/camera/image_raw',
                'output_topic': '/autospeed/detections'
            }]
        )
    ])
```

### Model Conversion Flow

#### Option 1: PyTorch Checkpoint
```bash
# You provide: best.pt
# Backend automatically creates:
#   1. best.onnx (PyTorch → ONNX conversion)
#   2. best.onnx.fp16.engine (ONNX → TensorRT engine)
```

#### Option 2: Pre-converted ONNX
```bash
# You provide: model.onnx
# Backend creates:
#   1. model.onnx.fp32.engine (ONNX → TensorRT engine)
```

## Output Format

The node publishes `vision_msgs::msg::Detection2DArray` with:

```cpp
Detection2DArray {
  header: std_msgs/Header
  detections: [
    {
      bbox: {
        center: {x, y}      // Bounding box center
        size_x: width       // Box width
        size_y: height      // Box height
      }
      results: [
        {
          hypothesis: {
            class_id: "0"   // Object class (as string)
            score: 0.95     // Confidence score
          }
        }
      ]
    },
    ...
  ]
}
```

All coordinates are in the **original image space** (not letterboxed).

## Preprocessing Details

### Letterbox Transformation

1. **Calculate scale**: `scale = min(640/width, 640/height)`
2. **Resize image**: `new_size = (orig_w * scale, orig_h * scale)`
3. **Create padded canvas**: 640x640 with gray (114, 114, 114)
4. **Paste resized image**: Center it with calculated padding
5. **Store parameters**: `scale`, `pad_x`, `pad_y` for later use

Example:
```
Original: 1920x1080
Scale: 640/1920 = 0.333
Resized: 640x360
Padding: pad_x=0, pad_y=140
Final: 640x640 (image in center, gray bars top/bottom)
```

## Post-Processing Details

### 1. Confidence Filtering
```cpp
for each prediction:
  if max_class_score < conf_threshold:
    discard
```

### 2. NMS (Non-Maximum Suppression)
```cpp
sort predictions by confidence (descending)
for each prediction:
  if not suppressed:
    keep it
    suppress all overlapping boxes (IoU > iou_threshold)
```

### 3. Coordinate Transformation
```cpp
// Transform from letterbox space to original image
x_orig = (x_letterbox - pad_x) / scale
y_orig = (y_letterbox - pad_y) / scale

// Clamp to image bounds
x_orig = clamp(x_orig, 0, orig_width)
y_orig = clamp(y_orig, 0, orig_height)
```

## Differences from Standard TensorRT Backend

| Aspect | Standard Backend | AutoSpeed Backend |
|--------|------------------|-------------------|
| **Preprocessing** | ImageNet normalization | Letterbox + /255 |
| | `(x - mean) / std` | `x / 255.0` |
| **Input Format** | ONNX only | PyTorch or ONNX |
| **Output** | Dense tensor (HxWxC) | Sparse predictions (Nx85) |
| **Post-processing** | Argmax or threshold | Confidence + NMS |
| **Coordinate System** | Simple resize | Letterbox transformation |
| **Use Case** | Segmentation, Depth | Object Detection |

## Dependencies

- **TensorRT** (≥8.0)
- **CUDA** (≥11.0)
- **OpenCV** (≥4.0)
- **ROS2 Humble/Iron**
- **vision_msgs** (for Detection2DArray)
- **PyTorch** (only needed for `.pt` → `.onnx` conversion)

## Troubleshooting

### Issue: "Failed to convert PyTorch to ONNX"
**Solution**: Ensure PyTorch is installed in your environment:
```bash
pip3 install torch torchvision
```

### Issue: Engine build is slow
**Reason**: TensorRT builds and optimizes the engine on first run. This is normal.
**Solution**: The engine is cached. Subsequent runs will be fast.

### Issue: Low detection accuracy
**Solutions**:
1. Adjust `conf_threshold` (try 0.5 or 0.4)
2. Use `fp32` instead of `fp16`
3. Check input image quality

### Issue: Too many overlapping boxes
**Solution**: Decrease `iou_threshold` (try 0.3 or 0.35)

## Performance Tips

1. **Use FP16 when possible**: 2-3x faster with minimal accuracy loss
2. **Cache engines**: Keep `.engine` files for faster startup
3. **Adjust thresholds**: Lower `conf_threshold` finds more objects but increases false positives
4. **Monitor FPS**: Use the built-in benchmark timer (prints every 100 frames)

## TODOs / Future Improvements

- [ ] Support batch inference (currently batch=1)
- [ ] Add per-class NMS
- [ ] Support dynamic input shapes
- [ ] Add visualization node
- [ ] Implement class-specific confidence thresholds

