# SodaSim ↔ Autoware 集成说明
# SodaSim ↔ Autoware Integration Notes

Soda.Sim 是一个基于 Unreal Engine 的仿真器，通过 ROS 2 发布车辆和传感器数据。本文档描述如何运行完整的 **VisionPilot 0.9 + SodaSim 1.4.0** 演示流水线。
Soda.Sim is an Unreal Engine–based simulator that publishes vehicle and sensor data over ROS 2. This document describes how to run the full **VisionPilot 0.9 + SodaSim 1.4.0** demo pipeline.

---

## 前置条件
## Prerequisites

- 从 [github.com/soda-auto/soda-sim/releases](https://github.com/soda-auto/soda-sim/releases) 下载 SodaSim 1.4.0
- ROS 2 Humble
- 按照说明构建的 VisionPilot 0.9，并已放置所需的模型文件
- 可获取 v4l2loopback 内核模块（`sudo apt install v4l2loopback-dkms`）
- SodaSim 1.4.0 downloaded from [github.com/soda-auto/soda-sim/releases](https://github.com/soda-auto/soda-sim/releases)
- ROS 2 Humble
- VisionPilot 0.9 built according to its instructions, with required model files in place
- v4l2loopback kernel module available (`sudo apt install v4l2loopback-dkms`)

---

## 快速开始
## Quick start

### 1. 加载虚拟设备
### 1. Load virtual device

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="SodaSim"
```

### 2. 启动 SodaSim 1.4.0
### 2. Launch SodaSim 1.4.0

启动 SodaSim。
Launch SodaSim.

打开 **主菜单 → 场景管理器**，选择 `Demo_AdasVehicle_Camera` 并点击**加载**。
Open **Main Menu → Scenario Manager**, select `Demo_AdasVehicle_Camera` and click **Load**.

点击顶部左侧工具栏中的**场景播放/停止**按钮开始发布。
Click the **Scenario Play/Stop** button in the top-left toolbar to start publishing.

### 3. 启动 GStreamer 桥接（ROS 2 → v4l2loopback）
### 3. Start the GStreamer bridge (ROS 2 → v4l2loopback)

```bash
# 首次构建一次，从 SodaSim/ros2_gstreamer/
# Build once (first time), from SodaSim/ros2_gstreamer/
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 运行（v4l2 模式）/ Run (v4l2 mode)
ros2 launch sodasim_gstreamer image_to_gstreamer.launch.py mode:=v4l2
```

这通过 GStreamer（UYVY 像素格式）将 `/vehicle/camera` 桥接到 `/dev/video10`。
This bridges `/vehicle/camera` → `/dev/video10` via GStreamer (UYVY pixel format).

### 4. 启动 VisionPilot 0.9
### 4. Launch VisionPilot 0.9

按照 `autoware.privately-owned-vehicles/VisionPilot/Production_Releases/0.9/` 中的说明构建 VisionPilot 0.9，然后：
Build VisionPilot 0.9 according to its own instructions in `autoware.privately-owned-vehicles/VisionPilot/Production_Releases/0.9/`, then:

```bash
visionpilot SodaSim/VisionPilot/visionpilot_sodasim.conf
```

预期输出：统一的 OpenCV 窗口，显示 EgoLanes 车道边界 + AutoSpeed CIPO 检测，带距离/速度叠加。
Expected output: unified OpenCV window showing EgoLanes lane boundaries + AutoSpeed CIPO detection with distance/speed overlay.

![Expected result](media/expected_result.gif)

---

## 相机配置
## Camera configuration

`Charge67_Autoware_VisionPilot` 车辆使用前向针孔相机，参数如下：
The `Charge67_Autoware_VisionPilot` vehicle uses a forward-facing pinhole camera with:

| 参数 / Parameter | 值 / Value |
|---|---|
| 分辨率 / Resolution | 1280 × 1060 |
| 水平视场角 / FOV (horizontal) | 50° |
| 相机高度 / Camera height | 0.90 m |
| 相机前向偏移 / Camera forward offset | 1.70 m |
| 俯仰角 / Pitch | 0° |
| ROS 2 话题 / ROS 2 topic | `/vehicle/camera` |
| 编码 / Encoding | `bgr8` / `ColorBGR8` |
| QoS | `BestEffort`, `KeepLast(1)`, `Volatile` |

VisionPilot 内部将帧裁剪为 1280 × 640（移除顶部 420 行），为 EgoLanes 模型输入提供干净的 2:1 宽高比（640 × 320）。
VisionPilot internally crops the frame to 1280 × 640 (top 420 rows removed), giving a clean 2:1 aspect ratio for the EgoLanes model input at 640 × 320.

要在 SodaSim 中查看或更改相机设置：选择自车 → `Open Vehicle Components` → 展开 `Camera Sensors` → 选择相机 → 编辑 `Publishing`（ROS 2）和 `Camera Sensor`（图像）参数。
To view or change camera settings in SodaSim: select the ego vehicle → `Open Vehicle Components` → expand `Camera Sensors` → select the camera → edit `Publishing` (ROS 2) and `Camera Sensor` (image) parameters.

![Camera publisher settings](media/camera_publisher_settings.png)

---

## 单应性标定
## Homography calibration

VisionPilot 的 `ObjectFinder` 使用 3×3 单应性矩阵 `H` 将边界框底部中心像素投影到真实世界地面平面坐标 `(X_forward, Y_lateral)`（单位：米）。
VisionPilot's `ObjectFinder` uses a 3×3 homography `H` to project bounding-box bottom-centre pixels to real-world ground-plane coordinates `(X_forward, Y_lateral)` in metres.

`Charge67_Autoware_VisionPilot` 相机设置的预标定矩阵位于 `SodaSim/VisionPilot/homography_sodasim.yaml`，由 `visionpilot_sodasim.conf` 引用。如果你更改相机安装位置、视场角或分辨率，则必须重新标定单应性矩阵。
A pre-calibrated matrix for the `Charge67_Autoware_VisionPilot` camera setup is provided at `SodaSim/VisionPilot/homography_sodasim.yaml` and referenced by `visionpilot_sodasim.conf`. If you change the camera mount position, FOV, or resolution, the homography must be recalibrated.

---

## 下一步
## Next steps

下一版本的 VisionPilot 和 SodaSim 将支持完整的闭环控制 — VisionPilot 的转向和纵向控制命令直接反馈到仿真车辆中。
The next version of VisionPilot and SodaSim will support full closed-loop control — VisionPilot steering and longitudinal commands fed back directly into the simulated vehicle.

---

## 联系方式
## Contact

- 邮箱 / Email: `sim@soda.auto`
- 仓库 / Repo: `https://github.com/soda-auto/soda-sim`
