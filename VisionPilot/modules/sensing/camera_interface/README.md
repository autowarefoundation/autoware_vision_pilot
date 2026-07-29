# V4L2 接口模块
# V4L2 Interface Module

## I. 概述
## I. Overview

本 **V4L2 接口** 模块提供了一个健壮的 C++ 包装器，用于从 V4L2（Video4Linux2）相机设备捕获视频帧并将其转换为 OpenCV `cv::Mat` 格式。
This **V4L2 Interface** module provides a robust C++ wrapper for capturing video frames from V4L2 (Video4Linux2) camera devices and converting them to OpenCV `cv::Mat` format.

本模块对于 VisionPilot 1.0 中的底层相机帧采集至关重要，具有以下特性：
This module is essential for low-level camera frame acquisition in VisionPilot 1.0, featuring:

- V4L2 相机捕获，接口直接连接到 V4L2 挂载设备（`/dev/video0`、`/dev/video1` 等）
- V4L2 camera capturing whose interface directly connects to V4L2 mounted devices (`/dev/video0`, `/dev/video1`, etc.)
- 无缝转换为 OpenCV `cv::Mat` 供下游处理
- Seamless conversion to OpenCV `cv::Mat` for downstream processing
- 互斥锁保护的线程安全帧和统计信息
- Mutex-protected thread-safe frame and statistics
- 能够指定期望的 FPS 和编解码器设置
- Able to specify desired FPS and codec settings
- 监控捕获、丢弃和错误等帧统计信息
- Monitor frames captured, dropped, and errors, etc.

## II. 架构与模块结构
## II. Architecture & module structure

### 1. 架构
### 1. Architecture

`V4L2Reader` 类是 V4L2 相机操作的主要接口。它遵循与 ROS2 相机订阅者类似的模式，但在更低的硬件级别运行。
The `V4L2Reader` class is the main interface for V4L2 camera operations. It follows a similar pattern to the ROS2 camera subscriber but operates at a lower hardware level.

基本上，对于 V4L2 捕获，我们使用简单、现成且有效的 [OpenCV 的 VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)：
Basically, for the V4L2 capture, we use a simple, off-the-shelf yet effective [OpenCV's VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) :

```cpp
cv::Mat frame;
camera_capture >> frame;  // 核心 OpenCV VideoCapture 操作 / Core OpenCV VideoCapture operation
```

这使用 OpenCV 的 VideoCapture 与 CAP_V4L2 后端进行直接 V4L2 设备访问。
This uses OpenCV's VideoCapture with the CAP_V4L2 backend for direct V4L2 device access.

```
┌─────────────────────────────────────────┐
│   Linux V4L2 Framework (/dev/videoX)    │
│   USB Video Device Driver               │
│   Camera Hardware                       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OpenCV VideoCapture (CAP_V4L2)         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   V4L2 接口模块 / V4L2 Interface Module  │
│  ┌─────────────────────────────────────┐│
│  │  V4L2Reader 类 / Class               ││
│  │  - get_latest_frame()                ││
│  │  - 统计与监控 / Statistics & monitoring ││
│  └─────────────────────────────────────┘│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   VisionPilot 应用层 / Application Layer │
└─────────────────────────────────────────┘
```

### 2. 模块结构
### 2. Module structure

```
v4l2_interface/
├── CMakeLists.txt
├── README.md
├── include/
│   └── v4l2_interface/
│       └── v4l2_reader.hpp
└── src/
    └── v4l2_reader.cpp
```

## III. 构建
## III. Build

### 1. 前置条件
### 1. Prerequisites

- ROS2（在 ROS2 Humble / Ubuntu 22.04 上测试）。
- ROS2 (tested on ROS2 Humble / Ubuntu 22.04).
    - `source /opt/ros/humble/setup.bash`
- 所需包：/ Required packages:
    - `OpenCV`

### 2. 步骤
### 2. Steps

```bash
# 1. 从根目录到发布目录 / From root to release directory
cd VisionPilot/development_releases/1.0

# 2. 创建构建目录 / Create build dir
mkdir -p build && cd build

# 3. 加载 ROS2（以防你忘了）/ Source ROS2 just in case you forgot to
source /opt/ros/humble/setup.bash

# 4. CMake 配置 / CMake configure
cmake ..

# 5. 编译 / Compile
make -j$(nproc)
```

### 3. 预期输出
### 3. Expected output

```bash
[ 97%] Building CXX object app/CMakeFiles/VisionPilot.dir/vision_pilot.cpp.o
[100%] Linking CXX executable ../VisionPilot
[100%] Built target VisionPilot
```

## IV. 测试
## IV. Test

本模块具有 V4L2 => OpenCV 流转换功能，因此需要 V4L2 图像流。
This module features V4L2 => OpenCV stream conversion, thus a V4L2 image stream is required.

你可以使用本地视频进行简单测试：
You can do a simple test using a local video by simply just:

1. 通过 [ffmpeg](https://ffmpeg.org/) 将其发布为 V4L2 流。
1. Publishing it as a V4L2 stream via [ffmpeg](https://ffmpeg.org/).

```bash
# 1. 安装包 / Install package
sudo apt update
sudo apt install ffmpeg -y
sudo apt install v4l2loopback-dkms -y

# 2. 加载模块（假设你将在 `/dev/video9` 流式传输）
# 2. Load the module (assuming you gonna stream it at `/dev/video9`)
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=9 card_label="Virtual Camera" exclusive_caps=1

# 3. 在该挂载发布循环视频 / Publish looping video at that mount
ffmpeg -re -stream_loop -1 -i <absolute path to local video> -f v4l2 -pix_fmt yuv420p /dev/video9
```

这样 `dev/video9` 处的 V4L2 流现已激活。你可以在另一个终端中使用以下命令查看它：
With this the V4L2 stream at `dev/video9` is now active. You can have a look at it from another terminal using:

```bash
ffplay /dev/video9
```

2. 然后，在另一个终端（或回到你构建 VisionPilot 的终端），我们可以在主 VisionPilot 循环中测试它：
2. Then, on another terminal (or back to the terminal that you built VisionPilot), we can test it inside the main VisionPilot loop:

```bash
# 第一个参数 `1` 表示以 V4L2 输入启动
# First argument `1` means starting this with V4L2 input
# 第二个参数是挂载的 V4L2 流
# Second argument being mounted V4L2 stream
# 第三个参数是期望的 FPS
# Third argument being desired FPS
./VisionPilot 1 /dev/video9 10
```
