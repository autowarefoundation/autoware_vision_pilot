# VisionPilot 部署（AGX Orin 离线安装）

[🇺🇸 English](README.md)

## 本目录中的文件

| 文件 | 用途 |
|------|---------|
| `setup.sh` | 主安装脚本 — 在 Orin 上以 root 身份运行 |

## 快速开始（在 Orin 上，有网络）

```bash
# 1. 通过 USB 将整个仓库传输到 Orin
scp -r vision_pilot/ user@orin-ip:~/

# 2. 在 Orin 上运行安装脚本
cd ~/vision_pilot/deploy
sudo bash setup.sh
```

## ONNX Runtime — GPU 版本（Jetson Orin/Xavier）

> **⚠️ 不要从源码编译 ORT GPU 版！** ORT v1.27+ 的 abseil 20250814 升级导致 nvcc 在 aarch64 上编译 `libonnxruntime_providers_cuda.so` 崩溃。预编译 wheel 完美避开此问题。

脚本从 **NVIDIA Jetson AI Lab** PyPI 索引安装 `onnxruntime-gpu`，该索引提供预编译的 aarch64 wheel，支持 CUDA + TensorRT。**不需要从源码编译。**

关键命令：
```bash
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install onnxruntime-gpu
```

## ONNX Runtime — CPU 版本（x86_64 测试）

在 x86_64 电脑上做 video 模式测试时，使用 CPU 版即可。没有 nvcc 编译问题。

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz
tar -xzf onnxruntime-linux-x64-1.28.0.tgz
cmake -DONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.28.0 -DGPU=OFF ..
```

## 前置条件

- NVIDIA JetPack 6.2.2（CUDA 12.6, TensorRT 8.6, cuDNN 8.9）
- Red Panda USB 适配器 + Linux 驱动（`modprobe panda`）
- GMSL 相机已连接（显示为 `/dev/video0` 或类似）

## 安装完成后

```bash
# 启动 CAN 接口
vp-can-up

# 运行 VisionPilot
VisionPilot

# 或带调试可视化
VisionPilot --debug-viz
```

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `can0 not found` | 运行 `sudo modprobe panda`，检查 USB 连接 |
| `Camera not found` | 检查 `ls /dev/video*`，确保 GMSL 线缆连接牢固 |
| `ONNX Runtime not found` | 验证 `/usr/share/visionpilot/onnxruntime/lib/` 是否存在 |
| `CUDAExecutionProvider not found` | 使用正确的 PIP_INDEX_URL 运行 `pip3 install onnxruntime-gpu` |
| `libonnxruntime.so not found` | 安装后运行 `sudo ldconfig` |
