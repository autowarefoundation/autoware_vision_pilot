# VisionPilot 0.5 - AutoSteer Production Release

This release enables autonomous steering using the EgoLanes and AutoSteer neural networks to detect lane lines determine
steering angle and navigate roads at a predetermined, desired speed.

This includes autonomous lane keeping with cruise control.

## C++ Inference Pipeline

Multi-threaded lane detection inference system with ONNX Runtime backend.

### Quick Start

[Download](https://github.com/microsoft/onnxruntime/releases) ONNX Runtime for the appropriate CUDA version and OS.

**Set ONNX Runtime path**

Unpack the ONNX runtime archive and set `ONNXRUNTIME_ROOT` to point to the directory as for example:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.22.0
```

_Note_: For Jetson AGX download appropriate ONNX runetime from [Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

**Build**

[Download](https://github.com/autowarefoundation/autoware.privately-owned-vehicles.git) VisionPilot source code.
Navigate to `VisionPilot/Production_Releases/0.5` subdirectory  which looks like:

```
0.5/
├── src/
│   ├── inference/          # Pure inference backend (no visualization)
│   │   ├── onnxruntime_session.cpp/hpp
│   │   ├── onnxruntime_engine.cpp/hpp
│   │   └── README.md
│   └── visualization/      # Visualization module (separate)
│       └── draw_lanes.cpp/hpp
├── scripts/                # Python utilities
├── main.cpp                # Multi-threaded pipeline
├── CMakeLists.txt          # Build configuration
└── run.sh                  # Runner script
```

and create `build` subdirectory:

```bash
mkdir -p build && cd build
```

**Build Options**

The pipeline supports two inference backends:

1. **ONNX Runtime (default)**: Uses ONNX Runtime with TensorRT execution provider
   ```bash
   cmake -DSKIP_ORT=OFF ../
   make -j$(nproc)
   ```
   Requires: `ONNXRUNTIME_ROOT` environment variable set

2. **TensorRT Direct (SKIP_ORT=ON)**: Uses TensorRT directly, bypassing ONNX Runtime
   ```bash
   cmake -DSKIP_ORT=ON ../
   make -j$(nproc)
   ```
   Requires: CUDA and TensorRT installed (searches common locations or set `TENSORRT_ROOT`)
   
   **Use this option when:**
   - Building on Jetson where ONNX Runtime GPU builds are problematic
   - You want to avoid ONNX Runtime dependency
   - You only need TensorRT inference

**Default Build (ONNX Runtime)**

```bash
cmake ../
make -j$(nproc)
cd ..
```

**Configure and Run**

```bash
# Edit run.sh to set paths and options
./run.sh
```

### Configuration (run.sh)

- `VIDEO_PATH`: Input video file
- `MODEL_PATH`: ONNX model (.onnx)
- `PROVIDER`: cpu or tensorrt (ignored when `SKIP_ORT=ON`, always uses TensorRT)
- `PRECISION`: fp32 or fp16 (TensorRT only)
- `DEVICE_ID`: GPU device ID
- `CACHE_DIR`: TensorRT engine cache directory
- `THRESHOLD`: Segmentation threshold (default: 0.0)
- `MEASURE_LATENCY`: Enable performance metrics
- `ENABLE_VIZ`: Enable visualization window
- `SAVE_VIDEO`: Save annotated output video
- `OUTPUT_VIDEO`: Output video path

**Note**: When building with `SKIP_ORT=ON`, the `PROVIDER` argument is ignored and TensorRT is always used directly.

### Performance

- **CPU**: 20-40ms per frame
- **TensorRT FP16**: 2-5ms per frame (200-500 FPS capable)

### Model Output

3-channel lane segmentation (320x640):
- Channel 0: Ego left lane (blue)
- Channel 1: Ego right lane (magenta)
- Channel 2: Other lanes (green)
