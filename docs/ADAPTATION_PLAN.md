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
| 操作系统 | JetPack 6.2.2 (Ubuntu 22.04 aarch64) |
| CUDA | 12.6 (JetPack 6.2.2自带) |
| TensorRT | 8.6 (JetPack 6.2.2自带) |
| cuDNN | 8.9 (JetPack 6.2.2自带) |
| 相机 | GMSL (通过MAX9296/MAX96712解串器连接) |
| CAN适配器 | Red Panda (USB连接，Linux内核驱动 `panda`) |
| 车型 | Lexus ES200 2023 (TSS2 CAN协议) |

---

## 2. 模块级适配清单

### 2.1 ONNX Runtime 引擎 (`modules/engine/`)

**现状**: 代码支持 `cpu`/`cuda`/`tensorrt` 三种provider，通过cmake变量 `ONNXRUNTIME_ROOT` 链接 `libonnxruntime.so`。

**需要做的事**:

| # | 任务 | 详情 | 风险 |
|--|------|------|------|
| 1 | 安装 aarch64 GPU 版 ONNX Runtime | NVIDIA Jetson AI Lab 提供预编译 wheel，`pip install onnxruntime-gpu` 即可。详见 INSTALL.md B4 节 | 低 — 一行命令搞定 |
| 2 | 验证 CUDA/TensorRT EP 可用 | `ort.get_available_providers()` 应包含 `CUDAExecutionProvider` + `TensorrtExecutionProvider` | 低 — 已验证 |
| 3 | 获取 ORT 头文件用于编译 | 从官方 aarch64 CPU tgz 中提取头文件 | 低 |

**不改代码**: `onnx_engine.cpp` 已完整支持 CUDA/TensorRT，无需修改。

> **⚠️ 重要：为什么不能从源码编译 ORT GPU 版**
>
> ORT 从 v1.27.0 开始内部升级了 abseil-cpp 到 20250814 版本。新版 `raw_hash_map.h` 使用了复杂的 C++ 模板写法，nvcc（CUDA 编译器）在 aarch64 架构上无法正确解析这些模板，导致编译 `libonnxruntime_providers_cuda.so` 时崩溃。
>
> ORT v1.28.0 release notes 明确写了已修复：`Built with abseil 20250814 under NVCC (#28586)`。
>
> **结论**：根本不需要从源码编译。NVIDIA Jetson AI Lab 提供了现成的预编译 wheel，直接 `pip install onnxruntime-gpu` 即可，完美避开这个坑。
>
> **CPU 用户**（x86_64 测试）：没有 nvcc 编译问题，直接从 GitHub releases 下载 `.tgz` 即可，详见 INSTALL.md B4-alt 节。

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
- 角度→扭矩P控制器 (`STEER_ANGLE_TO_TORQUE = 400.0`，保守起步，实车标定后可增至800)
- 优雅降级: CAN初始化失败时禁用CAN，程序继续运行

**需要验证/调参**:
- `STEER_ANGLE_TO_TORQUE = 400.0` — 保守起步，实车标定后可增至800（每步+100）
- `ACCEL_MAX = 2.0`, `ACCEL_MIN = -3.5` — 加速度限制
- `STEER_TORQUE_MAX = 1500.0` — 转向扭矩上限
- **CAN发送频率 ≤ 100Hz** — Red Panda和车辆ECU的CAN总线负载限制，发送过快会导致丢帧或总线错误

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

**ONNX Runtime**: 预编译 GPU wheel 来自 NVIDIA Jetson AI Lab：
```bash
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install onnxruntime-gpu
```
安装后验证 `ort.get_available_providers()` 包含 `CUDAExecutionProvider` 和 `TensorrtExecutionProvider`。


---

## 3. 适配流程 (按执行顺序)

### Phase 0: 环境准备 (Orin上)

```bash
# 1. 确认系统版本
cat /etc/nv_tegra_release    # 应显示 JetPack 6.x
nvidia-smi                    # 确认GPU可用
nvcc --version                # 确认CUDA

# 2. 锁定最大性能（编译和运行时都建议）
sudo jetson_clocks

# 3. 安装编译依赖
sudo apt update && sudo apt install -y \
    build-essential cmake git \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    coinor-libipopt-dev libcppad-dev liblapack-dev libblas-dev \
    nlohmann-json3-dev python3 python3-pip

# 3. 安装 ONNX Runtime GPU 版（一行命令）
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install onnxruntime-gpu
```

### Phase 1: ONNX Runtime 安装

```bash
# 从 NVIDIA Jetson AI Lab 安装预编译 GPU wheel（包含 CUDA + TensorRT）
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install onnxruntime-gpu

# 验证
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
assert 'CUDAExecutionProvider' in providers
assert 'TensorrtExecutionProvider' in providers
print('GPU ONNX Runtime OK!')
"

# 获取头文件（编译 VisionPilot 需要）
# ORT 的 .so 文件会被 pip 安装到 site-packages，头文件需要从官方 CPU tgz 获取
cd /tmp
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-aarch64-1.28.0.tgz
tar -xzf onnxruntime-linux-aarch64-1.28.0.tgz
sudo mkdir -p /usr/share/visionpilot/onnxruntime/lib /usr/share/visionpilot/onnxruntime/include

# 库文件来自 pip 安装
ORT_LIB=$(python3 -c "import onnxruntime,os; print(os.path.dirname(onnxruntime.__file__)+'/capi')")
sudo cp $ORT_LIB/libonnxruntime*.so* /usr/share/visionpilot/onnxruntime/lib/

# 头文件来自 CPU tgz
sudo cp -r onnxruntime-linux-aarch64-1.28.0/include/* /usr/share/visionpilot/onnxruntime/include/
echo '/usr/share/visionpilot/onnxruntime/lib' | sudo tee /etc/ld.so.conf.d/visionpilot.conf
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

### Phase 5: 测试运行（详见第7节分步验证计划）

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

**重要**：严格按第7节的Step1→Step5顺序执行，不要跳步。

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
- `can_interface.hpp`: `STEER_ANGLE_TO_TORQUE` 增益标定（400起步，每步+100）
- `vision_pilot.conf`: `speed_limit`, `Lf` 车辆参数

---

## 6. 风险与待确认事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GMSL相机不暴露为标准V4L2设备 | 相机无法使用 | 检查JetPack设备树，必要时写NvArgus包装器 |
| OpenCV V4L2后端在Orin上不可用 | 相机无法使用 | 从源码编译OpenCV，确保V4L2后端编译进去 |
| ONNX Runtime aarch64 GPU支持缺失 | 推理极慢 | 从 NVIDIA Jetson AI Lab pip 安装 GPU wheel
| H.yaml标定不准确 | 路径预测偏差大 | 精心标定，多场景验证 |
| CAN波特率不匹配 | 通信失败 | 确认Lexus TSS2使用500kbps |
| 转向P控制器增益不合适 | 控制振荡/不足 | 从400起步，每步+100，观察扭矩反馈，不超过1500 |

---

## 7. 分步验证计划（关键！）

**核心原则**：每个子系统独立验证通过后，再进入下一阶段。严禁跳步。

### Step 1: ONNX Runtime 验证（不涉及相机/CAN）

```bash
# 在Orin上，用任意一张图片测试推理引擎能否加载模型
# 1. 先用CPU模式跑通（最快）
./build/VisionPilot --mode video --video test.mp4 --engine cpu
# 2. 确认三个模型都加载成功，控制台无报错
# 3. 切换GPU模式
./build/VisionPilot --mode video --video test.mp4 --engine cuda
```

**通过标准**：模型加载成功，控制台输出推理耗时（<100ms/帧为正常）

### Step 2: 相机独立验证（不涉及CAN/推理）

```bash
# 1. 确认设备节点
ls /dev/video*
v4l2-ctl --list-devices

# 2. 确认支持的像素格式和分辨率
v4l2-ctl -d /dev/video0 --list-formats-ext

# 3. 用OpenCV直接采集一帧（Python脚本快速测试）
python3 -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
if ret:
    cv2.imwrite('test_frame.jpg', frame)
    print(f'OK: {frame.shape}')
else:
    print('FAIL')
cap.release()
"

# 4. 确认图像质量和方向
# 检查 test_frame.jpg 是否正确，无花屏/颜色异常
```

**通过标准**：成功采集图像，分辨率/帧率正确，图像无花屏

### Step 3: CAN独立验证（不涉及相机/推理）

```bash
# 1. 加载Red Panda驱动
sudo modprobe panda
ls /sys/class/net/can0    # 确认can0存在

# 2. 配置CAN波特率
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# 3. 用candump监听（车辆通电状态下）
timeout 5 candump can0 | head -20
# 应该能看到CAN帧，特别是0xAA轮速帧

# 4. 用cansend测试发送（谨慎！）
# 仅在方向盘未被锁死、周围安全时测试
# cansend can0 2E4#0000000000000000    # 发送零扭矩
```

**通过标准**：能收发CAN帧，无错误

### Step 4: 端到端联调（相机+推理+CAN，安全环境）

```bash
# 1. 先在video模式下跑完整pipeline（不发CAN）
./build/VisionPilot --mode video --video test.mp4

# 2. 检查可视化输出
# - BEV图像是否正确
# - 路径点是否合理
# - 加速度/转向角命令是否在预期范围

# 3. 切换到v4l2模式，接真实相机
./build/VisionPilot --mode v4l2

# 4. 监控CAN输出（只读模式）
candump can0 | grep -E "2E4|343"    # 观察发送的帧
```

**通过标准**：推理结果合理，可视化正常，CAN帧格式正确

### Step 5: 实车标定（安全封闭场地）

```bash
# 1. 先标定H矩阵（5m、10m、20m、30m各放一个标定点）
# 2. 从低速开始（<10km/h），有人在副驾随时接管
# 3. 调整STEER_ANGLE_TO_TORQUE增益（从400开始，每次+100）
# 4. 观察车辆响应：
#   - 增益太小：转向不足
#   - 增益太大：转向振荡
# 5. 记录最佳增益值
```

**通过标准**：低速下车辆能稳定跟踪规划路径

---

## 8. 前人踩坑经验（来自Autoware社区）

以下问题来自实际在Jetson上部署Autoware的用户分享的经验。虽然VisionPilot不直接使用Autoware，但底层依赖相似，值得借鉴。

### 8.1 OpenCV 版本冲突（最常见）

**现象**：编译报错，函数找不到或参数不匹配。

**原因**：JetPack可能自带OpenCV 4.0，但某些模块需要3.2.0的特定API。

**解决**：
```bash
# 检查当前OpenCV版本
pkg-config --modversion opencv4
# 如果是4.0且有冲突，降级：
sudo apt-get purge libopencv-dev
# 从NVIDIA源安装兼容版本
```

**对VisionPilot的影响**：VisionPilot使用现代OpenCV API，**兼容4.x**，通常不需要降级。但如果遇到编译错误，优先检查OpenCV版本。

### 8.2 CUDA 版本必须匹配

**现象**：运行时报错 `CUDA driver version is insufficient for CUDA runtime version`。

**原因**：编译时的CUDA版本和运行时的CUDA版本不一致。

**解决**：
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi    # 显示驱动支持的最高CUDA版本

# Autoware需要在cmake中指定CUDA版本
# 修改 autowarebuildflagextras.cmake 中的版本号
```

**对VisionPilot的影响**：JetPack 6.x自带CUDA 12.x，VisionPilot的ORT直接使用，**通常不需要手动指定版本**。

### 8.3 Eigen 版本问题（影响MPC编译）

**现象**：Ipopt/MPC模块编译报错，Eigen相关。

**原因**：系统可能有旧版Eigen，cmake找到旧版。

**解决**：
```bash
# 检查Eigen版本
dpkg -l | grep eigen
# 如果版本 < 3.3.7，需要升级
# 删除旧的cmake配置文件
sudo rm -rf /usr/lib/cmake/eigen3
# 安装新版本
sudo apt install libeigen3-dev
```

**对VisionPilot的影响**：VisionPilot的MPC模块依赖CppAD/Ipopt，间接依赖Eigen。**需要确认系统Eigen版本≥3.3.7**。

### 8.4 GStreamer 依赖（影响WebRTC和视频处理）

**现象**：WebRTC模块编译失败，或视频解码报错。

**原因**：缺少GStreamer开发包。

**解决**：
```bash
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad
```

**对VisionPilot的影响**：V4L2后端可能依赖GStreamer解码。**必须安装这些包**。

### 8.5 ARM 架构包名差异

**现象**：`apt install` 报错找不到包。

**原因**：某些包在aarch64上的名字与x86_64不同。

**解决**：使用 `apt search` 查找正确包名：
```bash
apt search opencv | grep arm
apt search eigen | grep arm
```

**对VisionPilot的影响**：需要逐个确认所有依赖包在aarch64上可用。

### 8.6 OpenCV 从源码编译（如果V4L2不工作）

**现象**：`cv::VideoCapture(0, cv::CAP_V4L2)` 打开失败或采集异常。

**原因**：预编译的OpenCV可能没有编译V4L2后端。

**解决**：从源码编译OpenCV，确保V4L2后端：
```bash
git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 4.5.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D CUDA_ARCH_BIN="8.7" \    # Orin的SM架构
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
```

**耗时**：约30-60分钟（取决于散热和编译核心数）。

**对VisionPilot的影响**：如果标准V4L2采集有问题，这是**必要的备选方案**。

### 8.7 编译散热注意事项

**现象**：编译过程中CPU降频，编译极慢或死机。

**原因**：Orin长时间满载会过热降频。

**解决**：
- 确保散热片/风扇安装到位
- 编译前运行 `sudo jetson_clocks` 锁定最大频率
- 分步编译，不要一次 `-j$(nproc)` 全开
- 监控温度：`tegrastats`

---

## 9. 交付物清单

### 已完成 (代码层面)
- [x] SocketCAN接口 (`can_interface.hpp/.cpp`)
- [x] 配置解析支持 `can_device` (`vision_pilot_config.hpp/.cpp`)
- [x] 主程序CAN集成 + 优雅降级 (`vision_pilot.cpp`)
- [x] 角度→扭矩P控制器 (400起步) (`can_interface.cpp`)
- [x] Toyota TSS2校验和/counter (`can_interface.cpp`)

### 待完成 (运行环境)
- [ ] ONNX Runtime aarch64 GPU版本 (编译或下载)
- [ ] Red Panda Linux驱动编译安装
- [ ] 相机标定 → H.yaml
- [ ] vision_pilot.conf 配置
- [ ] 分步验证通过（Step 1-5）

### 验证状态（实车调试时填写）
- [ ] Step 1: ORT推理验证 (video模式 + CPU → CUDA)
- [ ] Step 2: 相机采集验证 (V4L2 + OpenCV Python测试)
- [ ] Step 3: CAN收发验证 (candump + cansend)
- [ ] Step 4: 端到端联调 (v4l2模式 + 可视化)
- [ ] Step 5: 实车标定 (H矩阵 + P增益)
