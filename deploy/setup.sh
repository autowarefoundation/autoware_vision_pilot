#!/usr/bin/env bash
# VisionPilot offline setup script for NVIDIA AGX Orin (aarch64)
# Run as root on the Orin after transferring the deploy/ directory via USB.
#
# Usage:
#   sudo bash deploy/setup.sh
#
# Prerequisites:
#   - NVIDIA JetPack 6.x installed (provides CUDA 12.x, TensorRT, cuDNN)
#   - deploy/ directory copied to Orin (contains this script + optional tgz files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNXRUNTIME_VERSION="1.28.0"
INSTALL_PREFIX="/usr/share/visionpilot"
BUILD_DIR="${SCRIPT_DIR}/../VisionPilot/build"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo bash deploy/setup.sh)"
    exit 1
fi

# ── 1. System packages ─────────────────────────────────────────────────────
info "Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
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
    libblas-dev

# ── 2. CppAD / Ipopt header fix (Ubuntu 24.04 compat) ─────────────────────
info "Fixing CppAD Ipopt header symlink..."
hdr_dir="$(dirname "$(find /usr/include -name IpIpoptApplication.hpp 2>/dev/null | head -1)")"
if [[ -n "$hdr_dir" ]]; then
    ln -sf "$hdr_dir" /usr/include/coin-or
    info "  Symlinked /usr/include/coin-or -> $hdr_dir"
else
    warn "  IpIpoptApplication.hpp not found — Ipopt may not be installed"
fi

# ── 3. Python packages ─────────────────────────────────────────────────────
info "Installing Python packages..."
pip3 install --no-cache-dir --break-system-packages opencv-python numpy

# ── 4. CUDA / TensorRT / cuDNN (JetPack) ──────────────────────────────────
info "Checking JetPack CUDA/TensorRT..."
if dpkg -l | grep -q "cuda-toolkit"; then
    info "  CUDA toolkit detected"
else
    warn "  CUDA toolkit not found — ensure JetPack 6.x is installed"
fi
if dpkg -l | grep -q "libnvinfer"; then
    info "  TensorRT detected"
else
    warn "  TensorRT not found — ensure JetPack 6.x is installed"
fi
if dpkg -l | grep -q "libcudnn"; then
    info "  cuDNN detected"
else
    warn "  cuDNN not found — ensure JetPack 6.x is installed"
fi

# ── 5. ONNX Runtime ────────────────────────────────────────────────────────
info "Setting up ONNX Runtime ${ONNXRUNTIME_VERSION}..."
ORT_DIR="${INSTALL_PREFIX}/onnxruntime"

# Check if already installed
if [[ -d "$ORT_DIR" ]] && [[ -f "$ORT_DIR/lib/libonnxruntime.so" ]]; then
    info "  ONNX Runtime already installed at $ORT_DIR — skipping"
else
    # Method: pip install GPU wheel from NVIDIA Jetson AI Lab
    info "  Installing GPU ONNX Runtime from NVIDIA Jetson AI Lab..."
    export PIP_INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"
    pip3 install onnxruntime-gpu

    info "  Checking GPU providers..."
    python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('  Providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'CUDA provider not found!'
assert 'TensorrtExecutionProvider' in providers, 'TensorRT provider not found!'
print('  GPU ONNX Runtime OK!')
"

    # Copy .so files from pip install
    info "  Setting up library symlinks..."
    ORT_PIP_DIR="$(python3 -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__) + '/capi')")"
    mkdir -p "$ORT_DIR/lib" "$ORT_DIR/include"
    cp "$ORT_PIP_DIR"/libonnxruntime*.so* "$ORT_DIR/lib/"

    # Download CPU tgz for headers only
    info "  Fetching ORT headers from CPU tgz..."
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz" \
        -O /tmp/ort_headers.tgz
    tar -xzf /tmp/ort_headers.tgz -C /tmp/ort_headers
    cp -r /tmp/ort_headers/*/include/* "$ORT_DIR/include/"
    rm -rf /tmp/ort_headers.tgz /tmp/ort_headers

    info "  ONNX Runtime installed to $ORT_DIR"
fi

# ── 6. Red Panda CAN driver ────────────────────────────────────────────────
info "Setting up Red Panda CAN driver..."
if lsmod | grep -q panda; then
    info "  Red Panda driver already loaded"
elif modprobe panda 2>/dev/null; then
    info "  Red Panda driver loaded via modprobe"
else
    warn "  Red Panda driver not found (modprobe panda failed)"
    warn "  Install from: https://github.com/commaai/panda"
    warn "  Or build manually: git clone https://github.com/commaai/panda && cd panda/board && make"
fi

# Bring up CAN interface if Red Panda is connected
if ip link show can0 &>/dev/null; then
    info "  Bringing up can0 at 500k..."
    ip link set can0 down 2>/dev/null || true
    ip link set can0 type can bitrate 500000
    ip link set can0 up
    info "  can0 is UP"
else
    warn "  can0 not detected — plug Red Panda via USB and run:"
    warn "    sudo modprobe panda && sudo ip link set can0 type can bitrate 500000 && sudo ip link set can0 up"
fi

# ── 7. Build VisionPilot ───────────────────────────────────────────────────
info "Building VisionPilot..."
VP_SRC="${SCRIPT_DIR}/.."
if [[ ! -f "$VP_SRC/VisionPilot/CMakeLists.txt" ]]; then
    error "VisionPilot source not found at $VP_SRC/VisionPilot/CMakeLists.txt"
    error "Ensure the full repo was transferred, not just deploy/"
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DONNXRUNTIME_ROOT="$ORT_DIR" \
      -DGPU=ON \
      -DENABLE_ROS2_INTERFACE=OFF \
      -DCMAKE_CXX_STANDARD_LIBRARIES="-lcppad_lib" \
      "$VP_SRC/VisionPilot"

make -j"$(nproc)" VisionPilot

info "Build complete: $BUILD_DIR/VisionPilot"

# ── 8. Install binary + assets ─────────────────────────────────────────────
info "Installing VisionPilot..."
cp "$BUILD_DIR/VisionPilot" /usr/bin/VisionPilot
mkdir -p "$INSTALL_PREFIX/config" "$INSTALL_PREFIX/assets/icons" "$INSTALL_PREFIX/modules/models/weights"
cp -r "$VP_SRC/VisionPilot/config/"* "$INSTALL_PREFIX/config/"
cp -r "$VP_SRC/VisionPilot/assets/icons/"* "$INSTALL_PREFIX/assets/icons/" 2>/dev/null || true
cp -r "$VP_SRC/VisionPilot/modules/models/weights/"* "$INSTALL_PREFIX/modules/models/weights/" 2>/dev/null || true

# ── 9. Configure LD_LIBRARY_PATH for ONNX Runtime ─────────────────────────
info "Setting up library paths..."
cat > /etc/ld.so.conf.d/visionpilot.conf <<EOF
${ORT_DIR}/lib
EOF
ldconfig

# ── 10. Create CAN setup convenience script ────────────────────────────────
cat > /usr/local/bin/vp-can-up <<'SCRIPT'
#!/usr/bin/env bash
# Quick CAN interface setup for VisionPilot
sudo modprobe panda 2>/dev/null || echo "Warning: panda module not found"
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
echo "can0 is UP at 500kbps"
SCRIPT
chmod +x /usr/local/bin/vp-can-up

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
info "========================================="
info "  VisionPilot setup complete!"
info "========================================="
echo ""
info "Quick start:"
info "  1. Plug in Red Panda USB"
info "  2. Run:  vp-can-up"
info "  3. Run:  VisionPilot"
echo ""
info "If GMSL camera is not detected, check:"
info "  ls /dev/video*"
info "  v4l2-ctl --list-devices"
echo ""
info "Config file: $INSTALL_PREFIX/config/vision_pilot.conf"
