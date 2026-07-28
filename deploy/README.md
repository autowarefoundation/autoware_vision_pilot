# VisionPilot Deploy (Offline Installation for AGX Orin)

## Files in this directory

| File | Purpose |
|------|---------|
| `setup.sh` | Main install script — run on Orin as root |
| `onnxruntime-linux-aarch64-1.26.0.tgz` | ONNX Runtime for aarch64 (**download manually**, see below) |

## Quick Start (on the Orin)

```bash
# 1. Transfer this entire repo to the Orin via USB
scp -r deploy/ user@orin-ip:~/vision_pilot_deploy/

# 2. On the Orin, run the setup script
cd ~/vision_pilot_deploy
sudo bash setup.sh
```

## Manual ONNX Runtime Download

The ONNX Runtime `.tgz` is NOT included in the repo (~8MB). Download it before transfer:

**Option A: Download on your desktop**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-aarch64-1.26.0.tgz
```
Place the `.tgz` in the `deploy/` directory before transferring to the Orin.

**Option B: Download directly on the Orin**
If the Orin has internet access, `setup.sh` will auto-download it.

**Option C: OnnxRuntime from JetPack**
NVIDIA may ship ONNX Runtime with JetPack. Check:
```bash
dpkg -l | grep onnxruntime
```

## ONNX Runtime + GPU (CUDA/TensorRT) on Orin

The `onnxruntime-linux-aarch64-1.26.0.tgz` is the **CPU-only** package.
For GPU acceleration on Orin:

- CUDA and TensorRT are provided by **JetPack 6.x** (already installed).
- ONNX Runtime automatically discovers CUDA/TensorRT at runtime if they're in the system library path.
- To build ONNX Runtime from source with explicit GPU support:
  ```bash
  git clone --recursive https://github.com/microsoft/onnxruntime
  cd onnxruntime
  ./build.sh --config Release --use_cuda --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --build_shared_lib --skip_tests
  ```

## Prerequisites

- NVIDIA JetPack 6.x (CUDA 12.x, TensorRT 10.x, cuDNN 9.x)
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
| `libonnxruntime.so not found` | Run `sudo ldconfig` after install |
