# VisionPilot AGX Orin 安装与验证手册

> **目标**：在 AGX Orin 32G + GMSL Camera + Red Panda CAN 上部署 VisionPilot。
> **原则**：每一步都有命令、踩坑提醒、验证检查点。严格按顺序执行，不要跳步。
> **预计耗时**：首次约2-3小时（含编译），后续部署约30分钟。

---

## 目录

- [Part A: 代码传输](#part-a-代码传输)
- [Part B: Orin 环境搭建](#part-b-orin-环境搭建)
- [Part C: 相机验证](#part-c-相机验证)
- [Part D: CAN 验证](#part-d-can-验证)
- [Part E: ONNX Runtime 验证](#part-e-onnx-runtime-验证)
- [Part F: 编译 VisionPilot](#part-f-编译-visionpilot)
- [Part G: 端到端联调](#part-g-端到端联调)
- [Part H: 实车标定](#part-h-实车标定)
- [附录: 常见问题速查](#附录常见问题速查)

---

## Part A: 代码传输

### A1. 在电脑上准备代码

```bash
# 克隆仓库（含全部源码）
git clone -b dev https://github.com/feiyang2025/lexus.git vision_pilot
cd vision_pilot

# 确认目录结构完整
ls VisionPilot/CMakeLists.txt          # 应存在
ls VisionPilot/modules/engine/         # ONNX Runtime 引擎
ls VisionPilot/modules/models/weights/ # 6个ONNX模型文件
ls deploy/setup.sh                     # 安装脚本
ls config/vision_pilot.conf            # 配置文件
```

### A2. 传输到 Orin

```bash
# 方法1: SCP（Orin在同一网络）
scp -r vision_pilot/ user@<orin-ip>:~/

# 方法2: USB驱动器（离线环境，推荐）
# 插入U盘，将 vision_pilot/ 整个目录拷贝过去
```

**验证检查点 A**：Orin上 `ls ~/vision_pilot/VisionPilot/CMakeLists.txt` 应存在。

---

## Part B: Orin 环境搭建

以下所有命令都在 **Orin 上执行**。

### B1. 确认系统版本

```bash
cat /etc/nv_tegra_release
# 应显示 R36 (release 36.x) — 即 JetPack 6.x

nvidia-smi
# 应显示 GPU 可用，驱动版本 ≥ 540

nvcc --version
# 应显示 CUDA 12.6（JetPack 6.2.2自带）
```

> **踩坑**: 如果 `nvidia-smi` 找不到，说明 JetPack 没装好，需要重装。不要跳过这一步。

### B2. 锁定最大性能

Orin 长时间满载会过热降频，编译和运行时都建议锁定：

```bash
sudo jetson_clocks
```

> **踩坑**: 没有散热片/风扇的 Orin 编译时会降频，甚至死机。确保散热到位。

### B3. 安装编译依赖

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake git wget ca-certificates gnupg \
    python3 python3-pip \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-nice \
    libnice-dev \
    libsrtp2-dev \
    libboost-system-dev \
    nlohmann-json3-dev \
    coinor-libipopt-dev \
    libcppad-dev \
    liblapack-dev \
    libblas-dev \
    can-utils
```

> **踩坑 1**: `can-utils` 提供 `candump`/`cansend` 命令，用于测试CAN通信。如果 `apt` 报找不到包，运行 `apt search can-utils` 确认包名。

> **踩坑 2**: aarch64 上部分包名可能不同。如果某个包安装失败，用 `apt search <关键词>` 查找正确名称。

> **踩坑 3**: `libopencv-dev` 安装的是系统自带版本（通常4.x）。VisionPilot兼容4.x，一般没问题。如果后续编译报OpenCV相关错误，再考虑从源码编译（见[附录](#从源码编译-opencv)）。

### B4. 安装 ONNX Runtime（GPU版，直接 pip install）

**适用人群**：在 Jetson Orin/Xavier 上部署，需要 GPU 推理加速。

**说明**：NVIDIA 官方为 JetPack 6.x 提供了预编译的 `onnxruntime-gpu` wheel，包含完整 CUDA + TensorRT 支持。**不需要从源码编译**，一行命令搞定。

> **⚠️ 重要：为什么不能从源码编译 ORT GPU 版**
>
> 如果你尝试从源码编译 ONNX Runtime GPU 版（`libonnxruntime_providers_cuda.so`），会遇到 **abseil 20250814 + nvcc 模板错误**：
>
> - **原因**：ORT 从 v1.27.0 开始内部升级了 abseil-cpp 到 20250814 版本。新版 `raw_hash_map.h` 使用了复杂的 C++ 模板写法，nvcc（CUDA 编译器）在 aarch64 架构上无法正确解析这些模板，导致编译 `libonnxruntime_providers_cuda.so` 时崩溃。
> - **修复**：ORT v1.28.0 release notes 明确写了已修复：`Built with abseil 20250814 under NVCC (#28586)`。
> - **结论**：**根本不需要从源码编译**。NVIDIA Jetson AI Lab 提供了现成的预编译 wheel，直接 `pip install onnxruntime-gpu` 即可，完美避开这个坑。

```bash
# 1. 配置 NVIDIA Jetson AI Lab PyPI 源
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# 2. 安装 GPU 版 ONNX Runtime
pip3 install onnxruntime-gpu

# 3. 验证 GPU provider 可用
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'CUDA not found!'
assert 'TensorrtExecutionProvider' in providers, 'TensorRT not found!'
print('GPU ONNX Runtime OK!')
"

# 4. 找到 ORT 安装位置，设置 CMake 编译用的库路径
ORT_SITE=$(python3 -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")
sudo mkdir -p /usr/share/visionpilot/onnxruntime/lib /usr/share/visionpilot/onnxruntime/include
sudo cp "$ORT_SITE/capi/libonnxruntime"*.so* /usr/share/visionpilot/onnxruntime/lib/
```

> **踩坑 1**: 如果 `pip3 install onnxruntime-gpu` 报 404，确认 Orin 上的 JetPack 版本。不同 JetPack 版本对应不同 Index URL：
> - JetPack 6.1/6.2: `https://pypi.jetson-ai-lab.io/jp6/cu126`
> - JetPack 6.0: `https://pypi.jetson-ai-lab.io/jp6/cu122`
> - JetPack 5.x: 不支持（需要源码编译 ORT 1.17 以下版本）

> **踩坑 2**: 如果 `PIP_INDEX_URL` 不生效，可以手动下载 wheel 再安装：
> ```bash
> wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/<hash>/onnxruntime_gpu-<version>-cp310-cp310-linux_aarch64.whl
> pip3 install onnxruntime_gpu-*.whl
> ```
> 先访问 `https://pypi.jetson-ai-lab.io/jp6/cu126/` 查看可用版本列表。

> **踩坑 3**: 如果还是装不上，备选方案是下载 NVIDIA 论坛用户编译的 wheel：
> ```bash
> wget https://github.com/guyin24/onnxruntime-gpu-for-jetson/releases/download/v1.24.4/onnxruntime_gpu-1.24.4-cp310-cp310-linux_aarch64.whl
> pip3 install onnxruntime_gpu-1.24.4-cp310-cp310-linux_aarch64.whl
> ```

---

### B4-alt. 安装 ONNX Runtime（CPU版，用于 x86_64 测试）

**适用人群**：在 x86_64 电脑（Windows/Ubuntu）上做 video 模式测试，不需要 GPU 推理。

CPU 版没有 abseil + nvcc 的问题（因为没有 CUDA 编译），直接从 GitHub releases 下载即可：

```bash
# 1. 下载 CPU 版 ONNX Runtime（x86_64）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz

# 2. 解压
tar -xzf onnxruntime-linux-x64-1.28.0.tgz

# 3. 设置路径（用于 CMake 编译）
ORT_ROOT=$(pwd)/onnxruntime-linux-x64-1.28.0

# 4. 编译时指定
cmake -DONNXRUNTIME_ROOT="$ORT_ROOT" -DGPU=OFF ..
make -j VisionPilot
```

### B5. 安装 ONNX Runtime 开发依赖（编译必需）

VisionPilot 编译时还需要 ORT 的头文件。从官方 CPU 包中获取：

```bash
# 下载 CPU 版 tgz（仅用于头文件）
cd ~/vision_pilot
wget https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-aarch64-1.28.0.tgz
sudo tar -xzf onnxruntime-linux-aarch64-1.28.0.tgz -C /usr/share/visionpilot/
cd /usr/share/visionpilot
sudo mv onnxruntime-linux-aarch64-1.28.0 onnxruntime-headers

# 把头文件拷到 GPU 版目录
sudo cp -r onnxruntime-headers/include/* /usr/share/visionpilot/onnxruntime/include/
sudo rm -rf onnxruntime-headers
rm ~/vision_pilot/onnxruntime-linux-aarch64-1.28.0.tgz

# 配置动态链接库
echo "/usr/share/visionpilot/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/visionpilot.conf
sudo ldconfig
```

**验证检查点 B**：
```bash
ls /usr/share/visionpilot/onnxruntime/lib/libonnxruntime*.so*  # GPU .so 应存在
ls /usr/share/visionpilot/onnxruntime/include/onnxruntime_cxx_api.h  # 头文件应存在
pkg-config --modversion opencv4                                # 应显示 4.x
nvidia-smi                                                     # 应显示GPU可用
```

---

## Part C: 相机验证

在编译 VisionPilot 之前，先确认 GMSL 相机能正常采集图像。

### C1. 检查设备节点

```bash
ls /dev/video*
# 应显示 /dev/video0（或更多）
```

```bash
v4l2-ctl --list-devices
# 应显示相机设备名称和对应的 /dev/videoN
```

> **踩坑 1**: 如果没有 `/dev/video*`，检查 GMSL 线缆是否插好，相机是否通电。

> **踩坑 2**: GMSL 相机可能通过 Jetson VI (Video Input) 暴露为多个 `/dev/videoN`。通常第一个是主设备。

### C2. 确认像素格式和分辨率

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
# 确认支持 MJPEG 或 YUYV 或 UYVY
# 确认支持 1920x1080 或 1280x720
```

> **踩坑**: 如果只看到很低的分辨率（如 320x240），可能需要安装 GStreamer 插件或修改设备参数。

### C3. OpenCV Python 快速测试

```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
ret, frame = cap.read()
if ret:
    cv2.imwrite('/tmp/test_frame.jpg', frame)
    print(f'OK: {frame.shape}')
else:
    print('FAIL: 无法采集图像')
cap.release()
"
```

```bash
# 检查保存的图像
ls -la /tmp/test_frame.jpg
# 如果有图像，可以用 scp 拉到电脑上查看是否正确
```

> **踩坑 1**: 如果报错 `CAP_V4L2` 相关，可能是 OpenCV 没有编译 V4L2 后端。需要从源码编译 OpenCV（见[附录](#从源码编译-opencv)）。

> **踩坑 2**: 如果图像花屏或颜色异常，可能是像素格式不匹配。检查 `v4l2-ctl` 输出的格式，尝试在 Python 中手动指定。

> **踩坑 3**: GMSL 相机可能需要 `nvgstcapture` 或 `nvarguscamerasrc` 而非标准 V4L2。如果 OpenCV 无法采集，尝试 GStreamer 命令行测试：
> ```bash
> gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080,framerate=30/1 ! videoconvert ! autovideosink
> ```

**验证检查点 C**：`/tmp/test_frame.jpg` 存在且图像正确。

---

## Part D: CAN 验证

### D1. 加载 Red Panda 驱动

```bash
# 插入 Red Panda USB
lsusb | grep panda
# 应显示 panda 设备

# 加载内核模块
sudo modprobe panda
```

> **踩坑 1**: 如果 `modprobe panda` 失败，需要手动编译驱动：
> ```bash
> cd /tmp
> git clone https://github.com/commaai/panda
> cd panda/board
> make
> sudo insmod panda.ko
> ```

> **踩坑 2**: 确认 USB 口可用。Orin 有多个 USB 口，优先用 USB 3.0 口。

### D2. 配置 CAN 接口

```bash
# 确认 can0 存在
ip link show can0

# 配置波特率并启动
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# 确认状态
ip -details link show can0
# 应显示 state UP
```

> **踩坑**: 波特率 500000 (500kbps) 是 Lexus TSS2 的标准波特率。其他车型可能不同，但大部分丰田/雷克萨斯都是 500kbps。

### D3. 测试 CAN 收发

```bash
# 监听 CAN 帧（车辆通电状态下）
timeout 5 candump can0 | head -20
# 应该能看到 CAN 帧，特别是 0xAA（轮速）等 ID
```

> **踩坑**: 如果什么帧都看不到，确认：
> 1. 车辆电源已打开（ACC ON 或 READY）
> 2. Red Panda 的 CAN 接口线已连接到车辆 OBD-II 或 CAN 总线
> 3. 波特率正确（500kbps）

**验证检查点 D**：`candump` 能收到 CAN 帧。

---

## Part E: ONNX Runtime 验证

在编译 VisionPilot 之前，先确认 GPU 版 ONNX Runtime 已正确安装并能加载模型。

### E1. 验证 GPU Provider 可用

```bash
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'CUDA Missing!'
assert 'TensorrtExecutionProvider' in providers, 'TensorRT Missing!'
print('GPU ONNX Runtime OK!')
"
```

### E2. 验证 C++ 库链接

```bash
# 编译并运行一个小测试程序
ORT_ROOT="/usr/share/visionpilot/onnxruntime"
cat > /tmp/test_ort.cpp << 'EOF'
#include <iostream>
#include <onnxruntime_cxx_api.h>
int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    std::cout << "ONNX Runtime C++ API OK!" << std::endl;
    return 0;
}
EOF

g++ /tmp/test_ort.cpp -o /tmp/test_ort \
    -I"$ORT_ROOT/include" \
    -L"$ORT_ROOT/lib" -lonnxruntime

export LD_LIBRARY_PATH="$ORT_ROOT/lib:$LD_LIBRARY_PATH"
/tmp/test_ort
# 应输出: ONNX Runtime C++ API OK!
```

**验证检查点 E**：Python 显示 CUDA + TensorRT Provider，C++ 测试链接成功。

---

## Part F: 编译 VisionPilot

### F1. 配置 CMake

```bash
cd ~/vision_pilot/VisionPilot
mkdir -p build && cd build

ORT_ROOT="/usr/share/visionpilot/onnxruntime"

cmake -DONNXRUNTIME_ROOT="$ORT_ROOT" \
      -DGPU=ON \
      -DENABLE_ROS2_INTERFACE=OFF \
      ..
```

> **踩坑 1**: 如果 cmake 报找不到 `coinor-libipopt`，确认已安装：
> ```bash
> sudo apt install coinor-libipopt-dev
> ```

> **踩坑 2**: 如果 cmake 报找不到 `cppad`，确认已安装：
> ```bash
> sudo apt install libcppad-dev
> ```

> **踩坑 3**: cmake 输出中应显示 `ONNXRUNTIME_ROOT = /usr/share/visionpilot/onnxruntime`。如果不对，手动指定。

### F2. 编译

```bash
make -j$(nproc) VisionPilot
```

> **踩坑**: 编译可能需要 10-30 分钟（取决于散热和核心数）。如果编译极慢或死机，运行 `sudo jetson_clocks` 并减少并行数（如 `make -j4`）。

### F3. 验证编译产物

```bash
ls -la VisionPilot
# 应显示可执行文件

# 快速运行测试（video模式，用任意视频文件）
echo "编译成功！继续下一步验证。"
```

**验证检查点 F**：`build/VisionPilot` 可执行文件存在。

---

## Part G: 端到端联调

### G1. 用视频文件测试（不涉及相机/CAN）

```bash
cd ~/vision_pilot/VisionPilot/build

# 先用 video 模式测试整个 pipeline
# 需要一个测试视频文件（可以用 openpilot 的路试片段）
./VisionPilot --mode video --video /path/to/test.mp4
```

> **踩坑**: 如果没有测试视频，可以先跳过这步，直接进入 G2 用真实相机测试。

### G2. 用真实相机测试

```bash
# 先启动CAN（如果Red Panda已连接）
sudo modprobe panda 2>/dev/null
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up 2>/dev/null

# 运行 VisionPilot（v4l2模式）
./VisionPilot --mode v4l2
```

> **踩坑 1**: 第一次运行可能需要几秒钟加载模型，不要急着关掉。

> **踩坑 2**: 如果相机初始化失败，VisionPilot 应该会降级为无相机模式继续运行（检查控制台输出）。

> **踩坑 3**: 如果 CAN 初始化失败，VisionPilot 也会降级为无 CAN 模式继续运行。

### G3. 检查可视化输出

```bash
# 带调试可视化运行
./VisionPilot --debug-viz
```

**验证检查点 G**：
- 模型加载成功（控制台无报错）
- 如果有相机：图像正常显示
- 如果有CAN：candump 能看到发出的帧

---

## Part H: 实车标定

> **安全警告**：以下操作必须在封闭、安全的场地进行。副驾驶必须有人随时准备接管。从低速开始，逐步增加。

### H1. 标定单应性矩阵 H

这是最重要的标定步骤。H 矩阵决定了图像像素到世界坐标的映射。

```bash
# 1. 在道路上放置标定标记（5m、10m、20m、30m各一个）
# 2. 采集图像，记录标记的像素坐标
# 3. 运行标定脚本生成 H.yaml
cd ~/vision_pilot/VisionPilot
python3 scripts/find_homography_C_matrix.py --output config/H.yaml
```

> **踩坑**: 5m-30m 区域的标定精度直接影响路径预测。标定点越多、分布越广，结果越准确。

### H2. 低速测试（<10km/h）

```bash
# 1. 确保 CAN 已连接
vp-can-up    # 如果已安装，或手动执行 Part D 的命令

# 2. 运行 VisionPilot
./build/VisionPilot --mode v4l2

# 3. 从极低速开始（<10km/h），观察：
#   - 可视化窗口中的路径点是否在车道内
#   - 转向是否有响应（方向盘应该轻微转动）
#   - 是否有异常振动
```

### H3. 调整 P 控制器增益

```bash
# P增益在 VisionPilot/modules/sensing/vehicle_interface/include/vehicle_interface/can_interface.hpp
# 默认值: STEER_ANGLE_TO_TORQUE = 400.0
# 调整范围: 400 → 500 → 600 → 700 → 800
# 
# 观察标准：
#   - 增益太小：车辆转向不足，偏离车道
#   - 增益太大：方向盘抖动，扭矩输出剧烈波动
#   - 合适的增益：车辆能稳定跟随路径，方向盘平滑
```

> **踩坑**: 不要一次把增益调太大！从400开始，每次+100，每次调整后至少测试100米。

### H4. 记录最终参数

当找到最佳增益后，记录以下参数：

```bash
# 记录到配置文件
cat >> ~/vision_pilot/VisionPilot/config/vision_pilot.conf << 'EOF'

# ─── 实车标定参数 ──────────────────────────────────────────────
# STEER_ANGLE_TO_TORQUE = <你的最佳增益值>
# H.yaml = <你的标定结果>
EOF
```

**最终验证**：车辆能在低速下稳定跟踪规划路径，方向盘平滑，无异常振动。

---

## 附录: 常见问题速查

### Q1: 编译报错 `OpenCV not found`

```bash
# 检查OpenCV版本
pkg-config --modversion opencv4

# 如果版本不对，重装
sudo apt purge libopencv-dev
sudo apt install libopencv-dev
```

### Q2: 编译报错 `CUDA not found`

```bash
# 检查CUDA
nvcc --version
ls /usr/local/cuda/lib64/libcudart.so

# 确认环境变量
echo $CUDA_HOME
# 应为 /usr/local/cuda
```

### Q3: 运行时报错 `libonnxruntime.so not found`

```bash
# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/share/visionpilot/onnxruntime/lib:$LD_LIBRARY_PATH

# 或者永久配置
echo "/usr/share/visionpilot/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/visionpilot.conf
sudo ldconfig
```

### Q4: 相机图像花屏

```bash
# 检查像素格式
v4l2-ctl -d /dev/video0 --list-formats-ext

# 尝试指定格式
python3 -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# 尝试 YUYV
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
ret, frame = cap.read()
print('YUYV:', ret)
cap.release()
"
```

### Q5: CAN 无法通信

```bash
# 检查 Red Panda 是否被识别
lsusb | grep panda

# 检查内核模块
lsmod | grep panda

# 检查 CAN 接口状态
ip -details link show can0

# 重新配置
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

### Q6: 推理速度太慢

```bash
# 1. 确认是否用了 GPU provider
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 应该包含 CUDAExecutionProvider 和 TensorrtExecutionProvider

# 2. 如果只有 CPU，重新安装 GPU 版
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install --force-reinstall onnxruntime-gpu

# 3. GPU 加速后约 10-30ms/帧，远快于 CPU 的 200-500ms/帧
```

### 从源码编译 OpenCV（4.14.0，2026-07-18发布）

如果 V4L2 采集有问题，或需要 GStreamer 支持：

```bash
cd /tmp
git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 4.14.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D CUDA_ARCH_BIN="8.7" \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

> **耗时**: 约30-60分钟。确保散热良好，运行 `sudo jetson_clocks`。
