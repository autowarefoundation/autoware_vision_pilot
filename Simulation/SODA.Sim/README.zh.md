# SodaSim ↔ Autoware 集成说明

[🇺🇸 English](README.md)

Soda.Sim 是一个基于 Unreal Engine 的仿真器，通过 ROS 2 发布车辆和传感器数据。本文档描述如何运行完整的 **VisionPilot 0.9 + SodaSim 1.4.0** 演示流水线。

---

## 前置条件

- 从 [github.com/soda-auto/soda-sim/releases](https://github.com/soda-auto/soda-sim/releases) 下载 SodaSim 1.4.0
- ROS 2 Humble
- 按照说明构建的 VisionPilot 0.9，并已放置所需的模型文件
- 可获取 v4l2loopback 内核模块（`sudo apt install v4l2loopback-dkms`）

---

## 快速开始

### 1. 加载虚拟设备

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="SodaSim"
```

### 2. 启动 SodaSim 1.4.0

启动 SodaSim。

打开 **主菜单 → 场景管理器**，选择 `Demo_AdasVehicle_Camera` 并点击**加载**。

点击顶部左侧工具栏中的**场景播放/停止**按钮开始发布。

### 3. 启动 GStreamer 桥接（ROS 2 → v4l2loopback）

```bash
# 首次构建一次，从 SodaSim/ros2_gstreamer/
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 运行（v4l2 模式）
ros2 launch sodasim_gstreamer image_to_gstreamer.launch.py mode:=v4l2
```

这通过 GStreamer（UYVY 像素格式）将 `/vehicle/camera` 桥接到 `/dev/video10`。

### 4. 启动 VisionPilot 0.9

按照 `autoware.privately-owned-vehicles/VisionPilot/Production_Releases/0.9/` 中的说明构建 VisionPilot 0.9，然后：

```bash
visionpilot SodaSim/VisionPilot/visionpilot_sodasim.conf
```

预期输出：统一的 OpenCV 窗口，显示 EgoLanes 车道边界 + AutoSpeed CIPO 检测，带距离/速度叠加。

![Expected result](media/expected_result.gif)

---

## 相机配置

`Charge67_Autoware_VisionPilot` 车辆使用前向针孔相机，参数如下：

| 参数 | 值 |
|---|---|
| 分辨率 | 1280 × 1060 |
| 水平视场角 | 50° |
| 相机高度 | 0.90 m |
| 相机前向偏移 | 1.70 m |
| 俯仰角 | 0° |
| ROS 2 话题 | `/vehicle/camera` |
| 编码 | `bgr8` / `ColorBGR8` |
| QoS | `BestEffort`, `KeepLast(1)`, `Volatile` |

VisionPilot 内部将帧裁剪为 1280 × 640（移除顶部 420 行），为 EgoLanes 模型输入提供干净的 2:1 宽高比（640 × 320）。

要在 SodaSim 中查看或更改相机设置：选择自车 → `Open Vehicle Components` → 展开 `Camera Sensors` → 选择相机 → 编辑 `Publishing`（ROS 2）和 `Camera Sensor`（图像）参数。

![Camera publisher settings](media/camera_publisher_settings.png)

---

## 单应性标定

VisionPilot 的 `ObjectFinder` 使用 3×3 单应性矩阵 `H` 将边界框底部中心像素投影到真实世界地面平面坐标 `(X_forward, Y_lateral)`（单位：米）。

`Charge67_Autoware_VisionPilot` 相机设置的预标定矩阵位于 `SodaSim/VisionPilot/homography_sodasim.yaml`，由 `visionpilot_sodasim.conf` 引用。如果你更改相机安装位置、视场角或分辨率，则必须重新标定单应性矩阵。

---

## 下一步

下一版本的 VisionPilot 和 SodaSim 将支持完整的闭环控制 — VisionPilot 的转向和纵向控制命令直接反馈到仿真车辆中。

---

## 联系方式

- 邮箱：`sim@soda.auto`
- 仓库：`https://github.com/soda-auto/soda-sim`
