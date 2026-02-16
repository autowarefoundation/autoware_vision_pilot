# Middleware recipes for Vision Pilot

A modular, hardware and middleware-independent vision processing framework designed for autonomous vehicle perception tasks. VisionPilot provides a unified API for AI inference and visualization that can be deployed across different middleware systems.

## Architecture Philosophy

VisionPilot separates core AI processing from middleware-specific implementations, enabling seamless deployment across different robotic frameworks:

```raw
┌─────────────────────────────────────────────────────────┐
│                    MIDDLEWARE LAYER                     │
├─────────────────────┬───────────────────┬───────────────┤
│       ROS2          │      Zenoh        │    Future     │
│   Implementation    │  Implementation   │ Middlewares   │
│                     │                   │               │
│   ┌─────────────┐   │  ┌─────────────┐  │               │
│   │ ROS2 Nodes  │   │  │ Zenoh Nodes │  │      ...      │
│   │ & Topics    │   │  │ & Messages  │  │               │
│   └─────────────┘   │  └─────────────┘  │               │
└─────────────────────┴───────────────────┴───────────────┘
                       │                   
┌─────────────────────────────────────────────────────────┐
│                    COMMON LAYER                         │
│              (Framework Independent)                    │
├─────────────────────┬───────────────────┬───────────────┤
│  Inference Backends │ Visualization     │ Sensor Input  │
│                     │ Engines           │ Processing    │
│  • ONNX Runtime     │                   │               │
│  • TensorRT         │ • Segmentation    │ • Video       │
│  • Custom Backends  │ • Depth Maps      │ • Camera      │
│                     │ • Point Clouds    │ • Streaming   │
└─────────────────────┴───────────────────┴───────────────┘
```

## Key Benefits

### Middleware Independence

- **Write Once, Deploy Everywhere**: Core AI logic works across ROS2, Zenoh, and future middleware
- **No Vendor Lock-in**: Switch between middleware systems without changing AI code
- **Future Proof**: Add new middleware implementations without touching core engines

### Modular Design

- **Pluggable Backends**: Support for ONNX Runtime, TensorRT, and custom inference engines
- **Configurable Pipelines**: YAML-driven configuration for models, topics, and parameters
- **Independent Components**: Sensors, inference, and visualization can run independently
- **Performance Monitoring**: Built-in latency and FPS measurements
- **Concurrent Execution**: Multiple AI pipelines running simultaneously
- **Resource Efficient**: Shared common engines reduce memory footprint

## Current Implementations

### ROS2 (`/ROS2/`)

Production-ready ROS2 nodes providing:

- Native ROS2 topics and parameters
- Launch file orchestration
- Standard ROS2 message types
- Component-based node architecture

### Zenoh (`/Zenoh/`)

The Zenoh implementation providing:

- Edge computing deployments
- Low-latency communication
- Distributed AI processing
- Cloud-edge hybrid architectures

### Common Core (`/common/`)

Framework-agnostic engines providing:

- AI inference backends (ONNX Runtime, TensorRT)
- Visualization rendering (segmentation masks, depth maps)
- Sensor input processing (video streams, camera feeds)

## Supported Pipelines

### Segmentation

- **Scene Segmentation**: Binary foreground/background separation
- **Domain Segmentation**: Road/off-road classification  

### Depth Estimation

- **Scene 3D**: Monocular depth estimation

## Quick Start

### Download models

You need to create a folder for models and video first.

- Models: Download the models from [the Models folder](/Models).
- Video: You can get any dash cam video from YouTube.

```bash
cd VisionPilot/Middleware_Recipes
# Create folder
mkdir -p data && cd data
# Put your video into data, assuming its name is video.mp4
# Download the models
## Tool to download from Google Drive
pipx install gdown
## SceneSeg
gdown -O models/ 'https://docs.google.com/uc?export=download&id=1l-dniunvYyFKvLD7k16Png3AsVTuMl9f'
## Scene3D
gdown -O models/ 'https://docs.google.com/uc?export=download&id=19gMPt_1z4eujo4jm5XKuH-8eafh-wJC6'
## DomainSeg
gdown -O models/ 'https://docs.google.com/uc?export=download&id=1zCworKw4aQ9_hDBkHfj1-sXitAAebl5Y'
## AutoSpeed
gdown -O models/ 'https://docs.google.com/uc?export=download&id=1Zhe8uXPbrPr8cvcwHkl1Hv0877HHbxbB'
```

### ROS2 Implementation

- Install the dependencies in [ROS2](ROS2/README.md) first

```bash
cd VisionPilot/ROS2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select sensors models visualization \
  --cmake-args \
  -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
  -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# Run complete pipeline
ros2 launch models run_pipeline.launch.py \
  pipeline:=scene_seg \
  video_path:="../data/video.mp4"
```

### Zenoh Implementation

- Install the dependencies in [Zenoh](Zenoh/README.md) first

```bash
cd VisionPilot/Middleware_Recipes/Zenoh
# Set the environment variable
export LIBTORCH_INSTALL_ROOT=/path/to/libtorch/
export ONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-gpu-1.22.0
just all

# Run SceneSeg
just run_sceneseg
```

### Custom Middleware Integration

To add a new middleware implementation:

1. Create `/YourMiddleware/` directory
2. Implement thin wrapper nodes around common engines
3. Handle middleware-specific message passing
4. Leverage existing common backends and visualizers

## Repository Structure

```raw
VisionPilot/
├── common/         # Framework-agnostic core engines
├── ROS2/           # ROS2-specific implementation  
├── Zenoh/          # Zenoh-specific implementation
└── README.md       # This file
```

For technical details on core engines, see `common/README.md`
For ROS2-specific usage, see `ROS2/README.md`
For Zenoh-specific usage, see `Zenoh/README.md`
