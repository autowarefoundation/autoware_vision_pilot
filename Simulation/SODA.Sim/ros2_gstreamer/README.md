# SodaSim ROS 2 GStreamer 桥接
# SodaSim ROS 2 GStreamer Bridge

独立的 ROS 2 包，将 Soda.Sim ROS 2 相机话题（默认：`/vehicle/camera`）重新广播到 GStreamer 流水线中。
Standalone ROS 2 package that rebroadcasts the Soda.Sim ROS 2 camera topic (default: `/vehicle/camera`) into a GStreamer pipeline.

支持两种输出模式：
Two output modes are supported:

| 模式 / Mode | 用途 / Use case |
|------|----------|
| `udp` | 通过 UDP 流式传输 H.264/RTP — 使用 VLC 或任何支持 RTP 的播放器查看 / Stream H.264/RTP over UDP — view with VLC or any RTP-capable player |
| `v4l2` | 将原始帧写入 v4l2loopback 设备 — 馈入独立应用（如 VisionPilot 0.9）/ Write raw frames to a v4l2loopback device — feed into standalone apps (e.g. VisionPilot 0.9) |

## 依赖（Ubuntu 22.04）
## Dependencies (Ubuntu 22.04)

```bash
sudo apt install \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  v4l2loopback-dkms v4l2loopback-utils \
  ros-humble-cv-bridge ros-humble-image-transport
```

## 构建
## Build

```bash
source /opt/ros/humble/setup.bash
cd <SODASIM_REPO_ROOT>/ros2_gstreamer
unset CXX CC   # 避免过时的 ccache 环境变量 / avoid stale ccache env vars
colcon build --symlink-install
source install/setup.bash
```

## 启动 — UDP 模式（流式传输到网络）
## Launch — UDP mode (stream to network)

```bash
ros2 launch sodasim_gstreamer image_to_gstreamer.launch.py \
  mode:=udp host:=<VIEWER_IP> port:=5600
```

**VLC 验证：**
**VLC validation:**

创建 `test.sdp`：
Create `test.sdp`:
```sdp
v=0
o=- 0 0 IN IP4 0.0.0.0
s=ROS Camera
c=IN IP4 0.0.0.0
t=0 0
m=video 5600 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
a=recvonly
```

用 VLC 打开：`vlc test.sdp`
Open with VLC: `vlc test.sdp`

如果看不到视频，检查网络路径和防火墙（UDP 端口 5600）。
If you don't see video, check the network path and firewall (UDP port 5600).

## 启动 — v4l2 模式（VisionPilot 0.9 的虚拟相机）
## Launch — v4l2 mode (virtual camera for VisionPilot 0.9)

**1. 创建虚拟设备（每次启动一次）：**
**1. Create the virtual device (once per boot):**

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="SodaSim"
sudo chmod 666 /dev/video10
```

**2. 启动桥接：**
**2. Launch the bridge:**

```bash
ros2 launch sodasim_gstreamer image_to_gstreamer.launch.py \
  mode:=v4l2 device:=/dev/video10
```

**3. 在连接到 VisionPilot 之前测试：**
**3. Test before connecting to VisionPilot:**

```bash
# 视觉检查 / Visual check
ffplay /dev/video10

# GStreamer 检查 / GStreamer check
gst-launch-1.0 v4l2src device=/dev/video10 ! videoconvert ! autovideosink
```

**4. 配置 VisionPilot 0.9**（`visionpilot_sodasim.conf`）：
**4. Configure VisionPilot 0.9** (`visionpilot_sodasim.conf`):

```ini
mode=camera
source.camera.auto_select=false
source.camera.device_id=/dev/video10
```

## 所有启动参数
## All launch arguments

| 参数 / Argument | 默认值 / Default | 描述 / Description |
|----------|---------|-------------|
| `mode` | `udp` | 输出模式：`udp` 或 `v4l2` / Output mode: `udp` or `v4l2` |
| `input_topic` | `/vehicle/camera` | 要读取的 ROS2 图像话题 / ROS2 image topic to read from |
| `target_fps` | `30.0` | 目标输出帧率 / Target output framerate |
| `host` | `127.0.0.1` | [udp] 目标主机 / [udp] Destination host |
| `port` | `5600` | [udp] 目标 UDP 端口 / [udp] Destination UDP port |
| `device` | `/dev/video10` | [v4l2] V4L2 loopback 设备路径 / [v4l2] V4L2 loopback device path |
