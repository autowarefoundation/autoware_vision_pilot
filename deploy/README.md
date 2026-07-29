# VisionPilot Deploy (Offline Installation for AGX Orin)

[🇨🇳 中文版](README.zh.md)

## Files in this directory

| File | Purpose |
|------|---------|
| `setup.sh` | Main install script — run on Orin as root |

## Quick Start (on the Orin, with network)

```bash
# 1. Transfer this entire repo to the Orin via USB
scp -r vision_pilot/ user@orin-ip:~/

# 2. On the Orin, run the setup script
cd ~/vision_pilot/deploy
sudo bash setup.sh
```

## ONNX Runtime — GPU Version (Jetson Orin/Xavier)

> **⚠️ Do NOT compile ORT GPU from source!** ORT v1.27+ abseil 20250814 upgrade causes nvcc to crash on aarch64 when compiling `libonnxruntime_providers_cuda.so`. Pre-built wheel avoids this entirely.

The script installs `onnxruntime-gpu` from **NVIDIA Jetson AI Lab** PyPI index, which provides pre-built aarch64 wheels with CUDA + TensorRT support. **No source compilation needed.**

The key command:
```bash
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install onnxruntime-gpu
```

## ONNX Runtime — CPU Version (x86_64 Testing)

For video mode testing on x86_64 machines, use the CPU version. No nvcc compilation issues.

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz
tar -xzf onnxruntime-linux-x64-1.28.0.tgz
cmake -DONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.28.0 -DGPU=OFF ..
```

## Prerequisites

- NVIDIA JetPack 6.2.2 (CUDA 12.6, TensorRT 8.6, cuDNN 8.9)
- Red Panda USB adapter + Linux driver (`modprobe panda`)
- GMSL camera connected (appears as `/dev/video0` or similar)

## After Installation

```bash
# Start CAN interface
vp-can-up

# Run VisionPilot
VisionPilot

# Or with debug visualization
VisionPilot --debug-viz
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `can0 not found` | Run `sudo modprobe panda`, check USB connection |
| `Camera not found` | Check `ls /dev/video*`, ensure GMSL cable is secure |
| `ONNX Runtime not found` | Verify `/usr/share/visionpilot/onnxruntime/lib/` exists |
| `CUDAExecutionProvider not found` | Run `pip3 install onnxruntime-gpu` with correct PIP_INDEX_URL |
| `libonnxruntime.so not found` | Run `sudo ldconfig` after install |
