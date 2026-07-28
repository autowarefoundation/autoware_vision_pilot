# VisionPilot → AGX Orin 32G + GMSL Camera + Red Panda CAN 适配方案

## 0. 系统总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VisionPilot 数据流                          │
│                                                                     │
│  GMSL Camera ──→ V4L2 (/dev/video0) ──→ cv::Mat frame (BGR)       │
│       │                                                            │
│       ▼                                                            │
│  ImagePreprocessor                                                 │
│    ├─ warpPerspective(C) ──→ warped (BEV 1024×512)   → AutoDrive  │
│    └─ top-crop + resize ──→ resized (1024×512)        → AutoSteer  │
│                                                          AutoSpeed  │
│       │                                                            │
│       ▼                                                            │
│  InferencePipeline (3个ONNX模型并行推理)                            │
│    ├─ AutoDrive  (prev+curr ImageNet归一化)  → CIPO距离/曲率/标志  │
│    ├─ AutoSteer  (curr 0-1归一化)            → 64个路径点          │
│    └─ AutoSpeed  (curr letterbox)            → YOLO目标检测         │
│       │                                                            │
│       ▼                                                            │
│  Fusion (粒子滤波器)                                               │
│    ├─ LongitudinalFusion → CIPO距离/速度                           │
│    └─ LateralFusion → CTE/航向角/曲率                               │
│       │                                                            │
│       ▼                                                            │
│  Planner (CppAD/IPopt MPC)                                        │
│    ├─ LongitudinalPlanner → 加速度命令                              │
│    └─ LateralPlanner → 转向角序列                                  │
│       │                                                            │
│       ▼                                                            │
│  CAN Interface (SocketCAN)                                        │
│    ├─ read()  ← 0xAA 轮速 (m/s)                                   │
│    └─ write() → 0x2E4 转向扭矩 + 0x343 纵向控制                   │
│       │                                                            │
│       ▼                                                            │
│  Visualization (OpenCV窗口 / WebRTC)                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. 硬件环境

| 组件 | 规格 |
|------|------|
| 计算平台 | NVIDIA Jetson AGX Orin 32GB |
| 操作系统 | JetPack 6.x (Ubuntu 22.04 aarch64) |
| CUDA | 12.x (JetPack自带) |
| TensorRT | 10.x (JetPack自带) |
| cuDNN | 9.x (JetPack自带) |
| 相机 | GMSL (通过MAX9296/MAX96712解串器连接) |
| CAN适配器 | Red Panda (USB连接，Linux内核驱动 `panda`) |
| 车型 | Lexus ES200 2023 (TSS2 CAN协议) |

---

## 2. 模块级适配清单

### 2.1 ONNX Runtime 引擎 (`modules/engine/`)

**现状**: 代码支持 `cpu`/`cuda`/`tensorrt` 三种provider，通过cmake变量 `ONNXRUNTIME_ROOT` 链接 `libonnxruntime.so`。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|---|------|------|------|
| 1 | 下载/编译 aarch64 ONNX Runtime | 官方无aarch64 GPU包。选项: (A) JetPack自带的ORT (如有); (B) 从源码编译 `--use_cuda --use_tensorrt`; (C) 用CPU版本先跑通 | 中 — 编译耗时约30min |
| 2 | 验证 CUDA/TensorRT EP 可用 | Orin 上 `libonnxruntime.so` 需要能找到 `libcudart.so`, `libnvinfer.so`, `libcudnn.so` | 低 — JetPack已安装 |
| 3 | 模型兼容性 | FP32 ONNX模型在Orin上直接可用。INT8需要校准表 | 低 |

**不改代码**: `onnx_engine.cpp` 已完整支持 CUDA/TensorRT，无需修改。

---

### 2.2 相机接口 (`modules/sensing/camera_interface/`)

**现状**: `V4L2CameraInterface` 通过 OpenCV `VideoCapture(device_number, cv::CAP_V4L2)` 采集。配置MJPEG编码、目标FPS。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|---|------|------|------|
| 1 | 确认GMSL设备节点 | `ls /dev/video*` 和 `v4l2-ctl --list-devices`。GMSL通过Jetson VI可能出现在 `/dev/video0` 或 `/dev/video1` | 低 |
| 2 | 确认像素格式 | `v4l2-ctl -d /dev/video0 --list-formats-ext`。GMSL通常输出UYVY/YUYV，需要OpenCV能正确解码 | 中 — 可能需要V4L2后端支持 |
| 3 | 确认分辨率/帧率 | `v4l2-ctl -d /dev/video0 --list-framesizes`。设置为模型需要的1920×1080或1280×720 | 低 |
| 4 | 配置V4L2参数 | `vision_pilot.conf` 中设置 `source.v4l2_device` 和 `source.v4l2_fps` | 低 |
| 5 | 测试OpenCV V4L2后端 | 在Orin上编译OpenCV并验证 `cv::CAP_V4L2` 能正常工作 | 中 — Jetson OpenCV可能需要从源码编译 |

**不改代码**: `v4l2_camera_interface.cpp` 已实现完整的V4L2采集逻辑，无需修改代码。只需确认设备节点和配置参数。

**可能的问题**:
- GMSL相机可能通过 `nvgstcapture` 或 `nvarguscamerasrc` 而非标准V4L2暴露。如果是这样，需要额外写一个 `GmslCameraInterface` 继承 `CameraInterface`。
- 但大多数GMSL解串器(MAX9296/MAX96712)在JetPack上会注册为标准V4L2设备，OpenCV可以直接使用。

---

### 2.3 CAN接口 (`modules/sensing/vehicle_interface/`)

**现状**: `CanInterface` 通过 SocketCAN 读取轮速(0xAA)、发送转向扭矩(0x2E4)和纵向控制(0x343)。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|---|------|------|------|
| 1 | 安装Red Panda Linux驱动 | `git clone https://github.com/commaai/panda && cd panda && make`。模块名: `panda` | 低 |
| 2 | 加载驱动 | `sudo modprobe panda` → 产生 `can0` 接口 | 低 |
| 3 | 配置CAN波特率 | `sudo ip link set can0 type can bitrate 500000 && sudo ip link set can0 up` | 低 |
| 4 | 验证CAN通信 | `candump can0` 观察0xAA轮速帧。确认数据格式与DBC一致 | 中 — 需要车辆通电 |
| 5 | 权限配置 | CAN socket需要root权限。创建udev规则或加入 `plugdev` 组 | 低 |
| 6 | 适配CAN ID | 当前代码写死0xAA/0x2E4/0x343，与Lexus ES TSS2一致。验证实际CAN总线 | 低 |

**代码已做好的事** (`can_interface.cpp`):
- SocketCAN socket创建、绑定、超时设置
- 0xAA轮速解码 (4轮平均，kph→m/s)
- 0x2E4转向扭矩 + Toyota XOR校验和 + counter
- 0x343纵向控制 + Toyota XOR校验和 + counter
- 角度→扭矩P控制器 (`STEER_ANGLE_TO_TORQUE = 800.0`)
- 优雅降级: CAN初始化失败时禁用CAN，程序继续运行

**需要验证/调参**:
- `STEER_ANGLE_TO_TORQUE = 800.0` — P控制器增益需要在实车上标定
- `ACCEL_MAX = 2.0`, `ACCEL_MIN = -3.5` — 加速度限制
- `STEER_TORQUE_MAX = 1500.0` — 转向扭矩上限

---

### 2.4 标定文件 (`config/`)

**现状**: `H.yaml` 是Zenseact Open Dataset的相机→世界单应性矩阵。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|---|------|------|------|
| 1 | 相机内参标定 | 用棋盘格标定GMSL相机的焦距、畸变系数 | 中 |
| 2 | 相机外参标定 | 安装GMSL相机后，标定相机相对车辆的位置和姿态 | 中 |
| 3 | 单应性矩阵H | 在实际道路上采集标定点，计算像素→世界的映射 | 高 — 核心参数 |
| 4 | 预处理矩阵C | 运行 `find_homography_C_matrix.py` 从H生成C | 低 — 脚本已有 |
| 5 | 验证BEV效果 | 在实际道路上检查warped图像的BEV投影是否正确 | 中 |

**关键文件**:
- `config/H.yaml` — 需要用实际相机参数替换
- `scripts/find_homography_C_matrix.py` — 自动从H生成C矩阵
- 构建时CMake会运行此脚本，生成 `build/config/homography_C_matrix.yaml`

---

### 2.5 模型权重 (`modules/models/weights/`)

**现状**: 已包含6个ONNX文件 (3模型 × 2精度)。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|---|------|------|------|
| 1 | 确认模型文件完整 | `ls -la modules/models/weights/`，检查文件大小 | 低 |
| 2 | 选择精度 | FP32安全但慢，INT8快但需要校准。Orin 32G算力足够FP32 | 低 |
| 3 | 模型兼容性测试 | 在Orin上运行一次推理，确认ORT能正确加载模型 | 低 |

**不改代码**: 模型加载逻辑 (`inference.cpp:find_model()`) 已支持本地路径和系统路径。

---

### 2.6 配置文件 (`config/vision_pilot.conf`)

**现状**: 默认配置使用 `source.mode = video`。

**需要修改为**:

```ini
# ─── 输入源 ──────────────────────────────────────────────────────
source.mode             = v4l2

source.v4l2_device      = /dev/video0     # GMSL设备节点
source.v4l2_fps         = 10              # 目标帧率

# ─── CAN总线 (Red Panda) ─────────────────────────────────────────
source.can_device       = can0            # SocketCAN接口

# ─── ONNX模型引擎 ────────────────────────────────────────────────
engine.provider         = cuda            # cuda 或 tensorrt
engine.device_id        = 0
engine.cache_dir        = /tmp/visionpilot_trt_cache
engine.workspace_gb     = 1.0

# ─── 模型精度 ────────────────────────────────────────────────────
model.precision         = fp32            # fp32 或 int8

# ─── 车辆参数 (Lexus ES200) ──────────────────────────────────────
speed_limit             = 33.3            # m/s (120 km/h)
Lf                      = 2.67            # 前轴到质心距离 (m)

# ─── 可视化 ──────────────────────────────────────────────────────
visualization_on        = true
webrtc_on               = false
webrtc_port             = 8080
```

---

### 2.7 依赖项安装

**Orin上需要的apt包** (aarch64):

```bash
# 编译工具链
build-essential cmake git

# OpenCV
libopencv-dev

# GStreamer (WebRTC)
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
libgstreamer-plugins-bad1.0-dev

# 数学库
coinor-libipopt-dev libcppad-dev liblapack-dev libblas-dev

# JSON
nlohmann-json3-dev

# 网络/WebRTC
libnice-dev libsrtp2-dev libboost-system-dev

# Python (构建脚本)
python3 python3-pip
pip3 install opencv-python numpy
```

**ONNX Runtime**: 需要aarch64版本。来源:
1. JetPack 6.x 可能自带 (检查 `dpkg -l | grep onnxruntime`)
2. 从源码编译 (`git clone --recursive https://github.com/microsoft/onnxruntime && ./build.sh --use_cuda --use_tensorrt`)
3. NVIDIA NGC容器内已包含

---

## 3. 适配流程 (按执行顺序)

### Phase 0: 环境准备 (Orin上)

```bash
# 1. 确认系统版本
cat /etc/nv_tegra_release    # 应显示 JetPack 6.x
nvidia-smi                    # 确认GPU可用
nvcc --version                # 确认CUDA

# 2. 安装编译依赖
sudo apt update && sudo apt install -y \
    build-essential cmake git \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    coinor-libipopt-dev libcppad-dev liblapack-dev libblas-dev \
    nlohmann-json3-dev python3 python3-pip

# 3. 检查ONNX Runtime
dpkg -l | grep onnxruntime
# 如果没有，需要编译或下载aarch64版本
```

### Phase 1: ONNX Runtime 安装

```bash
# 方案A: 从源码编译 (推荐，确保GPU支持)
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release \
    --use_cuda --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --build_shared_lib --skip_tests -j$(nproc)

# 编译产物在 build/Linux/Release/
sudo cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

### Phase 2: Red Panda CAN 驱动

```bash
# 1. 编译驱动
git clone https://github.com/commaai/panda
cd panda
make

# 2. 加载模块
sudo insmod panda.ko    # 或 modprobe panda (如果安装到modules)

# 3. 配置CAN
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# 4. 验证
candump can0    # 应该能看到0xAA等CAN帧
```

### Phase 3: 编译 VisionPilot

```bash
# 在Orin上
cd /path/to/vision_pilot/VisionPilot
mkdir build && cd build

cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DGPU=ON \
      -DENABLE_ROS2_INTERFACE=OFF \
      ..

make -j$(nproc) VisionPilot
```

### Phase 4: 标定与配置

```bash
# 1. 替换 H.yaml (用实际相机标定结果)
cp /path/to/your/H.yaml config/H.yaml

# 2. 编辑 vision_pilot.conf
# 设置 source.mode=v4l2, source.v4l2_device, source.can_device

# 3. 运行预处理脚本生成C矩阵
python3 scripts/find_homography_C_matrix.py --output build/config/homography_C_matrix.yaml
```

### Phase 5: 测试运行

```bash
# 1. 先测试相机
ls /dev/video*
v4l2-ctl -d /dev/video0 --list-formats-ext

# 2. 先测试CAN
candump can0 | head -20

# 3. 运行VisionPilot
cd build
./VisionPilot

# 4. 带调试可视化
./VisionPilot --debug-viz
```

---

## 4. 不需要修改的模块

以下模块代码已经完全适配，**不需要任何修改**:

| 模块 | 原因 |
|------|------|
| `engine/` (ONNX Runtime) | 已支持CUDA/TensorRT，跨平台 |
| `models/` (AI推理) | 纯ONNX推理，平台无关 |
| `fusion/` (粒子滤波) | 纯数学计算，平台无关 |
| `planning/` (MPC规划) | CppAD/IPopt，平台无关 |
| `visualization/` (显示) | OpenCV + GStreamer，跨平台 |
| `debug/` (调试视图) | 纯OpenCV绘图 |
| `logging/` (日志) | printf宏 |
| `common/` (工具函数) | OpenCV + filesystem |
| `config/` (配置解析) | 纯C++ |
| `camera_interface/` (V4L2) | 已实现V4L2采集，无需改代码 |
| `vehicle_interface/` (CAN) | 已实现SocketCAN，无需改代码 |
| `image_preprocessing/` | OpenCV warpPerspective，跨平台 |

---

## 5. 需要修改的文件 (最小集合)

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `config/vision_pilot.conf` | 改 `source.mode=v4l2`，设置设备节点 | 必须 |
| `config/H.yaml` | 用实际相机标定结果替换 | 必须 |
| `deploy/setup.sh` | 安装脚本，apt+ORT+驱动 | 建议 |

**可选调参**:
- `can_interface.hpp`: `STEER_ANGLE_TO_TORQUE` 增益标定
- `vision_pilot.conf`: `speed_limit`, `Lf` 车辆参数

---

## 6. 风险与待确认事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GMSL相机不暴露为标准V4L2设备 | 相机无法使用 | 检查JetPack设备树，必要时写NvArgus包装器 |
| OpenCV V4L2后端在Orin上不可用 | 相机无法使用 | 从源码编译OpenCV，确保V4L2后端编译进去 |
| ONNX Runtime aarch64 GPU支持缺失 | 推理极慢 | 从源码编译ORT with CUDA/TensorRT |
| H.yaml标定不准确 | 路径预测偏差大 | 精心标定，多场景验证 |
| CAN波特率不匹配 | 通信失败 | 确认Lexus TSS2使用500kbps |
| 转向P控制器增益不合适 | 控制振荡/不足 | 从小增益开始，逐步增大，观察扭矩反馈 |

---

## 7. 交付物清单

### 已完成 (代码层面)
- [x] SocketCAN接口 (`can_interface.hpp/.cpp`)
- [x] 配置解析支持 `can_device` (`vision_pilot_config.hpp/.cpp`)
- [x] 主程序CAN集成 + 优雅降级 (`vision_pilot.cpp`)
- [x] 角度→扭矩P控制器 (`can_interface.cpp`)
- [x] Toyota TSS2校验和/counter (`can_interface.cpp`)

### 待完成 (运行环境)
- [ ] ONNX Runtime aarch64 GPU版本 (编译或下载)
- [ ] Red Panda Linux驱动编译安装
- [ ] 相机标定 → H.yaml
- [ ] vision_pilot.conf 配置
- [ ] 端到端测试
