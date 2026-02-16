#!/usr/bin/env bash
set -ex

# ==============================================================================
# 1. PATHS & SETTINGS
# ==============================================================================

# CUDA and cuDNN locations
CUDA_HOME="/usr/local/cuda-12.6"
CUDNN_HOME="/usr/local/cuda-12.6"

# TensorRT location
TENSORRT_HOME="/usr/lib/aarch64-linux-gnu"

# GPU Architecture (90 = H100/Hopper, 87 = Jetson Orin)
CUDA_ARCH="87"

# Source Control
ONNXRUNTIME_BRANCH="main"

# ==============================================================================
# 2. SETUP WORKSPACE
# ==============================================================================

WORK_ROOT="$HOME/onnxruntime-gpu"
INSTALL_DIR="${WORK_ROOT}/install"
REPO_DIR="${WORK_ROOT}/onnxruntime"


echo "Building in: ${WORK_ROOT}"
mkdir -p ${WORK_ROOT}
cd ${WORK_ROOT}

# Clone
if [ ! -d "${REPO_DIR}" ]; then
    git clone https://github.com/microsoft/onnxruntime ${REPO_DIR}
    cd ${REPO_DIR}
    git checkout ${ONNXRUNTIME_BRANCH} || echo "Branch not found, using default"
else
    cd ${REPO_DIR}
fi

git submodule update --init --recursive

# ==============================================================================
# 3. DETECT PARALLELISM
# ==============================================================================

detect_cuda_max_jobs() {
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    local cicc_mem_per_job=6
    local safe_jobs=$(( (total_ram_gb - 8) / cicc_mem_per_job ))

    if [ $safe_jobs -lt 2 ]; then safe_jobs=2; fi
    if [ $safe_jobs -gt 12 ]; then safe_jobs=12; fi 
    echo $safe_jobs
}

ARCH=$(uname -m)
if [[ "${ARCH}" = "aarch64" ]] || [[ "${ARCH}" = "arm64" ]]; then
    MAX_JOBS=$(detect_cuda_max_jobs)
    export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
    export MAKEFLAGS="-j${MAX_JOBS}"
    echo "ARM64 Detected: Limiting to ${MAX_JOBS} parallel jobs"
else
    export MAX_JOBS=8
    export CMAKE_BUILD_PARALLEL_LEVEL=8
    export MAKEFLAGS="-j8"
fi

# ==============================================================================
# 4. BUILD COMMAND
# ==============================================================================

# Note: Removed CMAKE_CUDA_FLAGS and cccl includes as requested.
# Reverted CMAKE_CXX_FLAGS to original style.

./build.sh \
    --config Release \
    --update \
    --parallel \
    --build \
    --build_shared_lib \
    --skip_tests \
    --skip_submodule_sync \
    --allow_running_as_root \
    --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I${CUDA_HOME}/include" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    --cmake_extra_defines onnxruntime_ENABLE_CONTRIB_OPS=OFF \
    --cmake_extra_defines onnxruntime_USE_GSL=OFF \
    --cuda_home ${CUDA_HOME} \
    --cudnn_home ${CUDNN_HOME} \
    --use_tensorrt \
    --tensorrt_home ${TENSORRT_HOME}

# ==============================================================================
# 5. INSTALL
# ==============================================================================

cd build/Linux/Release
make install

echo "--------------------------------------------------------"
echo "Build Complete!"
echo "Libs:    ${INSTALL_DIR}/lib"
echo "Headers: ${INSTALL_DIR}/include"
echo "--------------------------------------------------------"