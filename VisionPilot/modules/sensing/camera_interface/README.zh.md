# V4L2 接口模块

[🇺🇸 English](README.md)

## I. 概述

本 **V4L2 接口** 模块提供了一个健壮的 C++ 包装器，用于从 V4L2（Video4Linux2）相机设备捕获视频帧并将其转换为 OpenCV `cv::Mat` 格式。

本模块对于 VisionPilot 1.0 中的底层相机帧采集至关重要，具有以下特性：

- V4L2 相机捕获，接口直接连接到 V4L2 挂载设备（`/dev/video0`、`/dev/video1` 等）
- 无缝转换为 OpenCV `cv::Mat` 供下游处理
- 互斥锁保护的线程安全帧和统计信息
- 能够指定期望的 FPS 和编解码器设置
- 监控捕获、丢弃和错误等帧统计信息

## II. 架构与模块结构

### 1. 架构

`V4L2Reader` 类是 V4L2 相机操作的主要接口。它遵循与 ROS2 相机订阅者类似的模式，但在更低的硬件级别运行。

基本上，对于 V4L2 捕获，我们使用简单、现成且有效的 [OpenCV 的 VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)：

```cpp
cv::Mat frame;
camera_capture >> frame;
```

这使用 OpenCV 的 VideoCapture 与 CAP_V4L2 后端进行直接 V4L2 设备访问。

```
┌─────────────────────────────────────────┐
│   Linux V4L2 Framework (/dev/videoX)    │
│   USB Video Device Driver               │
│   CameraHardware                       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OpenCV VideoCapture (CAP_V4L2)         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   V4L2 接口模块                         │
│  ┌─────────────────────────────────────┐│
│  │  V4L2Reader 类                      ││
│  │  - get_latest_frame()               ││
│  │  - 统计与监控                       ││
│  └─────────────────────────────────────┘│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   VisionPilot 应用层                    │
└─────────────────────────────────────────┘
```

### 2. 模块结构

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

### 1. 前置条件

- ROS2（在 ROS2 Humble / Ubuntu 22.04 上测试）。
    - `source /opt/ros/humble/setup.bash`
- 所需包：
    - `OpenCV`

### 2. 步骤

```bash
# 1. 从根目录到发布目录
cd VisionPilot/development_releases/1.0

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 加载 ROS2（以防你忘了）
source /opt/ros/humble/setup.bash

# 4. CMake 配置
cmake ..

# 5. 编译
make -j$(nproc)
```

### 3. 预期输出

```bash
[ 97%] Building CXX object app/CMakeFiles/VisionPilot.dir/vision_pilot.cpp.o
[100%] Linking CXX executable ../VisionPilot
[100%] Built target VisionPilot
```

## IV. 测试

本模块具有 V4L2 => OpenCV 流转换功能，因此需要 V4L2 图像流。

你可以使用本地视频进行简单测试：

1. 通过 [ffmpeg](https://ffmpeg.org/) 将其发布为 V4L2 流。

```bash
# 1. 安装包
sudo apt update
sudo apt install ffmpeg -y
sudo apt install v4l2loopback-dkms -y

# 2. 加载模块（假设你将在 `/dev/video9` 流式传输）
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=9 card_label="Virtual Camera" exclusive_caps=1

# 3. 在该挂载发布循环视频
ffmpeg -re -stream_loop -1 -i <absolute path to local video> -f v4l2 -pix_fmt yuv420p /dev/video9
```

这样 `dev/video9` 处的 V4L2 流现已激活。你可以在另一个终端中使用以下命令查看它：

```bash
ffplay /dev/video9
```

2. 然后，在另一个终端（或回到你构建 VisionPilot 的终端），我们可以在主 VisionPilot 循环中测试它：

```bash
# 第一个参数 `1` 表示以 V4L2 输入启动
# 第二个参数是挂载的 V4L2 流
# 第三个参数是期望的 FPS
./VisionPilot 1 /dev/video9 10
```
