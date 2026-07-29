# 相机订阅者模块

[🇺🇸 English](README.md)

## I. 概述

本相机订阅者模块是一个基于 ROS2 的中间件组件，执行以下功能：

1. 订阅和监听从任何 ROS2 图像源（如 CARLA 等仿真器、硬件相机、rosbag 回放等）暴露的 ROS2 话题（例如 `sensor_msgs/image`）。
2. 将接收到的图像消息转换为 OpenCV 格式 `cv::Mat` 对象进行处理。
3. 维护具有时间关键缓冲区的线程安全队列，以确保低延迟处理。

本模块用于闭环测试期间的 ROS2-OpenCV 图像桥接。

其他功能：

- 流状态跟踪，确认 ROS2 图像消息流是否已启动
- 接收帧的验证，确认它们是否可用且已被正确读取
- 能够处理多种编码，支持 RGB、Mono 等多种图像格式
- 还有一些不错的订阅健康统计（帧接收、丢弃、错误指标等）

## II. 架构与模块结构

### 1. 架构

```
ROS2 发布者（相机、仿真器、rosbag 回放等）
            ↓
            ↓
    [ROS2 图像话题]
            ↓
            ↓
ROS2ImageSubscriber（相机订阅者模块）
            |
            |---- [is_stream_started] 标志：流是否活跃？
            |---- [is_valid_frame]    标志：帧是否有效？
            │
            |---- [cv_bridge 转换]
            │
            |---- [线程安全队列 - 大小 = 1（时间关键）]
            │
            |---- [帧元数据跟踪]
            ↓
            ↓
OpenCV cv::Mat 流
            ↓
            ↓
VisionPilot 流水线（E2E 模型推理和其他处理）
```

### 2. 模块结构

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

### 1. 前置条件

- ROS2（在 ROS2 Humble / Ubuntu 22.04 上测试）。
    - `source /opt/ros/humble/setup.bash`
- 所需包：
    - `rclcpp`
    - `sensor_msgs`
    - `cv_bridge`
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

本模块具有 ROS2 => OpenCV 流转换功能，因此需要发布图像的 ROS2 话题。

你可以使用本地视频进行简单测试：

1. 通过 [`ros-<distro>-image-publisher`](https://docs.ros.org/en/humble/p/image_publisher/) 将其发布为 ROS2 图像消息。

```bash
# 1. 安装包，假设你使用 ROS2 Humble
sudo apt update
sudo apt install ros-humble-image-publisher

# 2. 建立 ROS2 话题（假设话题为 `/camera/image`）
ros2 run image_publisher image_publisher_node --ros-args \
-p filename:="<video absolute path>" \
-p publish_rate:=30.0 \
-p frame_id:="camera_link" \
-r image_raw:=/camera/image
```

这样话题 `/camera/image` 现已激活并从该本地视频发布帧，作为 ROS2 消息。

2. 然后，使用提供的 `camera_viewer_node` 可执行文件可视化从桥接接收到的 `cv::Mat`。

```bash
# 1. 从根目录到辅助目录
cd VisionPilot/development_releases/auxiliaries

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 加载 ROS2（以防你忘了）
source /opt/ros/humble/setup.bash

# 4. CMake 配置
cmake ..

# 5. 编译
make -j$(nproc)

# 6. 执行
./camera_viewer_node /camera/image 1
```
