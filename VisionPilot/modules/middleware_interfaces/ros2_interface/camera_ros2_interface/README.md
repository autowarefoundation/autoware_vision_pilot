# 相机订阅者模块
# Camera Subscriber Module

## I. 概述
## I. Overview

本相机订阅者模块是一个基于 ROS2 的中间件组件，执行以下功能：
This camera subscriber module is a ROS2-based middleware component that does these following:

1. 订阅和监听从任何 ROS2 图像源（如 CARLA 等仿真器、硬件相机、rosbag 回放等）暴露的 ROS2 话题（例如 `sensor_msgs/image`）。
1. Subscribing and listening to an exposed ROS2 topic (for example, `sensor_msgs/image`) from any ROS2 image source (simulators like CARLA, or hardware cameras, rosbag playback, etc.).
2. 将接收到的图像消息转换为 OpenCV 格式 `cv::Mat` 对象进行处理。
2. Transforming received image messages into OpenCV format `cv::Mat` objects for processing.
3. 维护具有时间关键缓冲区的线程安全队列，以确保低延迟处理。
3. Maintaining a thread-safe queue with time-critical buffer to ensure low-latency processing.

本模块用于闭环测试期间的 ROS2-OpenCV 图像桥接。
This module is intended for ROS2-OpenCV image bridge used during closed-loop testing.

其他功能：
Other features:

- 流状态跟踪，确认 ROS2 图像消息流是否已启动
- Stream status tracking which confirms whether the ROS2 image message stream has been started.
- 接收帧的验证，确认它们是否可用且已被正确读取
- Validation of received frames, whether they are available and have been read appropriately.
- 能够处理多种编码，支持 RGB、Mono 等多种图像格式
- Able to handle multiple encodings, supporting multiple image formats like RGB, Mono, etc.
- 还有一些不错的订阅健康统计（帧接收、丢弃、错误指标等）
- Also some nice statistics of subscription health (frame reception, drop, error metrics etc.).

## II. 架构与模块结构
## II. Architecture & module structure

### 1. 架构
### 1. Architecture

```
ROS2 发布者（相机、仿真器、rosbag 回放等）
ROS2 publisher (camera, simulator, rosbag replay etc.)
            ↓
            ↓
    [ROS2 图像话题 / ROS2 image topic]
            ↓
            ↓
ROS2ImageSubscriber（相机订阅者模块）
ROS2ImageSubscriber (camera subscriber module)
            |
            |---- [is_stream_started] 标志：流是否活跃？/ flag : stream active?
            |---- [is_valid_frame]    标志：帧是否有效？/ flag : frame valid?
            │
            |---- [cv_bridge 转换 / conversion]
            │
            |---- [线程安全队列 - 大小 = 1（时间关键）/ Thread-safe queue - size = 1 (time critical)]
            │
            |---- [帧元数据跟踪 / Frame metadata tracking]
            ↓
            ↓
OpenCV cv::Mat 流
OpenCV cv::Mat stream
            ↓
            ↓
VisionPilot 流水线（E2E 模型推理和其他处理）
VisionPilot pipeline (E2E models inference and other processing)
```

### 2. 模块结构
### 2. Module structure

```
camera_ros2_interface/
├── CMakeLists.txt
├── README.md
├── include/
│   └── camera_ros2_interface/
│       └── camera_ros2_interface.hpp
└── src/
    └── camera_ros2_interface.cpp
```

## III. 构建
## III. Build

### 1. 前置条件
### 1. Prerequisites

- ROS2（在 ROS2 Humble / Ubuntu 22.04 上测试）。
- ROS2 (tested on ROS2 Humble / Ubuntu 22.04).
    - `source /opt/ros/humble/setup.bash`
- 所需包：/ Required packages:
    - `rclcpp`
    - `sensor_msgs`
    - `cv_bridge`
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

本模块具有 ROS2 => OpenCV 流转换功能，因此需要发布图像的 ROS2 话题。
This module features ROS2 => OpenCV stream conversion, thus a ROS2 topic publishing images is required.

你可以使用本地视频进行简单测试：
You can do a simple test using a local video by simply just:

1. 通过 [`ros-<distro>-image-publisher`](https://docs.ros.org/en/humble/p/image_publisher/) 将其发布为 ROS2 图像消息。
1. Publishing it as ROS2 image messages via [`ros-<distro>-image-publisher`](https://docs.ros.org/en/humble/p/image_publisher/).

```bash
# 1. 安装包，假设你使用 ROS2 Humble
# 1. Install package, assuming you are using ROS2 Humble
sudo apt update
sudo apt install ros-humble-image-publisher

# 2. 建立 ROS2 话题（假设话题为 `/camera/image`）
# 2. Establish ROS2 topic (assuming the topic is `/camera/image`)
ros2 run image_publisher image_publisher_node --ros-args \
-p filename:="<video absolute path>" \
-p publish_rate:=30.0 \
-p frame_id:="camera_link" \
-r image_raw:=/camera/image
```

这样话题 `/camera/image` 现已激活并从该本地视频发布帧，作为 ROS2 消息。
With this the topic `/camera/image` is now active and publishing frames from that local video, as ROS2 messages.

2. 然后，使用提供的 `camera_viewer_node` 可执行文件可视化从桥接接收到的 `cv::Mat`。
2. Then, use the provided `camera_viewer_node` executable to visualize the `cv::Mat` received from the bridge.

```bash
# 1. 从根目录到辅助目录 / From root to auxiliaries directory
cd VisionPilot/development_releases/auxiliaries

# 2. 创建构建目录 / Create build dir
mkdir -p build && cd build

# 3. 加载 ROS2（以防你忘了）/ Source ROS2 just in case you forgot to
source /opt/ros/humble/setup.bash

# 4. CMake 配置 / CMake configure
cmake ..

# 5. 编译 / Compile
make -j$(nproc)

# 6. 执行 / Execute
./camera_viewer_node /camera/image 1
```
