#!/bin/bash
# EgoLanes Production Pipeline Runner

set -euo pipefail

# ============================================================================
# Environment Setup
# ============================================================================
# Set these paths for your system
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.22.0
export RERUN_SDK_ROOT=/path/to/rerun_cpp_sdk

# Add ONNX Runtime to library path
export LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}
export GST_DEBUG=1

# ============================================================================
# Pipeline Configuration
# ============================================================================

# ===== Mode Selection =====
MODE="video"              # "camera" or "video"

# ===== Required Parameters =====
VIDEO_PATH="/path/to/your/video.mp4"
MODEL_PATH="/path/to/Egolanes_fp32.onnx"
PROVIDER="tensorrt"       # Execution provider: 'cpu' or 'tensorrt'
PRECISION="fp16"          # Precision: 'fp32' or 'fp16' (for TensorRT)

# ===== ONNX Runtime Options =====
DEVICE_ID="0"             # GPU device ID (TensorRT only)
CACHE_DIR="/path/to/trt_cache"   # TensorRT engine cache directory

# ===== Steering Controller Parameters =====
KP="0.33"                 # Proportional gain
KI="0.01"                 # Integral gain
KD="-0.40"                # Derivative gain
KS="-0.3"                 # Curvature feedforward gain
CSV_LOG_PATH="metrics_kp${KP}_ki${KI}_kd${KD}_ks${KS}.csv"

# ===== AutoSteer Options (optional - temporal steering prediction) =====
ENABLE_AUTOSTEER="true"   # Enable AutoSteer temporal steering prediction
AUTOSTEER_MODEL_PATH="/path/to/AutoSteer_FP32.onnx"

# ===== CAN Interface Options (optional - ground truth) =====
ENABLE_CAN="false"        # Enable CAN interface for ground truth
CAN_INTERFACE="can0"      # CAN interface name (e.g., can0, can1) or .asc file path

# ===== Pipeline Options =====
THRESHOLD="0.0"           # Segmentation threshold
MEASURE_LATENCY="true"    # Enable latency metrics
ENABLE_VIZ="true"         # Enable OpenCV visualization
SAVE_VIDEO="false"        # Enable saving output video (requires ENABLE_VIZ=true)
OUTPUT_VIDEO="output_egolanes_${PRECISION}_${PROVIDER}.mp4"

# ===== Rerun Options (optional - requires RERUN_SDK_ROOT set and -DENABLE_RERUN=ON build) =====
ENABLE_RERUN="true"       # Enable Rerun logging and visualization
RERUN_SPAWN="true"        # Spawn live viewer window (RECOMMENDED - streams data directly, no RAM buffering)
RERUN_SAVE="false"        # Also save to .rrd file (⚠ buffers ALL data in RAM until completion!)
RERUN_PATH="egolanes.rrd" # Path to save .rrd file (if RERUN_SAVE=true)

# ============================================================================
# Validation
# ============================================================================

if [ "$MODE" != "camera" ] && [ "$MODE" != "video" ]; then
    echo "Error: MODE must be 'camera' or 'video'" >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH" >&2
    exit 1
fi

if [ "$MODE" == "video" ]; then
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "Error: Video file not found: $VIDEO_PATH" >&2
        exit 1
    fi
fi

if [ "$ENABLE_AUTOSTEER" == "true" ]; then
    if [ ! -f "$AUTOSTEER_MODEL_PATH" ]; then
        echo "Error: AutoSteer model file not found: $AUTOSTEER_MODEL_PATH" >&2
        exit 1
    fi
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "========================================"
echo "EgoLanes Inference Pipeline"
echo "========================================"
echo "Mode: $MODE"
if [ "$MODE" == "video" ]; then
    echo "Video: $VIDEO_PATH"
else
    echo "Camera: Interactive selection"
fi
echo "Model: $MODEL_PATH"
echo "Provider: $PROVIDER | Precision: $PRECISION"
if [ "$PROVIDER" == "tensorrt" ]; then
    echo "Device ID: $DEVICE_ID | Cache: $CACHE_DIR"
fi
echo "----------------------------------------"
echo "Steering Control:"
echo "  Kp: $KP | Ki: $KI | Kd: $KD | Ks: $KS"
echo "  Log: $CSV_LOG_PATH"
echo "----------------------------------------"
if [ "$ENABLE_AUTOSTEER" == "true" ]; then
    echo "AutoSteer: ENABLED"
    echo "  Model: $AUTOSTEER_MODEL_PATH"
    echo "  Provider: $PROVIDER | Precision: $PRECISION"
    echo "----------------------------------------"
fi
if [ "$ENABLE_CAN" == "true" ]; then
    echo "CAN Interface: ENABLED"
    echo "  Interface: $CAN_INTERFACE"
    echo "----------------------------------------"
fi
echo "Threshold: $THRESHOLD"
echo "OpenCV Viz: $ENABLE_VIZ"
echo "Save Video: $SAVE_VIDEO"
if [ "$SAVE_VIDEO" == "true" ]; then
    echo "  Output: $OUTPUT_VIDEO"
fi
echo "Rerun: $ENABLE_RERUN"
if [ "$ENABLE_RERUN" == "true" ]; then
    echo "  Spawn Viewer: $RERUN_SPAWN"
    if [ "$RERUN_SAVE" == "true" ]; then
        echo "  Save to file: $RERUN_PATH"
    fi
    if [ -z "$RERUN_SDK_ROOT" ]; then
        echo "  ⚠ WARNING: RERUN_SDK_ROOT not set!"
        echo "    Rerun may not work if not compiled with -DENABLE_RERUN=ON"
    fi
fi
echo "========================================"
echo ""

# ============================================================================
# Build Command
# ============================================================================

CMD="./build/egolanes_pipeline $MODE"

if [ "$MODE" == "video" ]; then
    CMD="$CMD \"$VIDEO_PATH\""
fi

CMD="$CMD \"$MODEL_PATH\" $PROVIDER $PRECISION $DEVICE_ID \"$CACHE_DIR\" $THRESHOLD $MEASURE_LATENCY $ENABLE_VIZ $SAVE_VIDEO \"$OUTPUT_VIDEO\""

# Add Steering Control params
CMD="$CMD --steering-control --Kp $KP --Ki $KI --Kd $KD --Ks $KS --csv-log \"$CSV_LOG_PATH\""

# Add AutoSteer flag
if [ "$ENABLE_AUTOSTEER" == "true" ]; then
    CMD="$CMD --autosteer \"$AUTOSTEER_MODEL_PATH\""
fi

# Add CAN interface flag
if [ "$ENABLE_CAN" == "true" ]; then
    CMD="$CMD --can-interface \"$CAN_INTERFACE\""
fi

# Add Rerun flags
if [ "$ENABLE_RERUN" == "true" ]; then
    if [ "$RERUN_SPAWN" == "true" ] && [ "$RERUN_SAVE" == "true" ]; then
        # Both spawn viewer AND save to file
        CMD="$CMD --rerun --rerun-save \"$RERUN_PATH\""
    elif [ "$RERUN_SPAWN" == "true" ]; then
        # Only spawn viewer (recommended - no memory buffering!)
        CMD="$CMD --rerun"
    elif [ "$RERUN_SAVE" == "true" ]; then
        # Only save to file (⚠ buffers ALL data in RAM until completion!)
        CMD="$CMD --rerun-save \"$RERUN_PATH\""
    fi
fi

# ============================================================================
# Run
# ============================================================================

if [ "$ENABLE_VIZ" == "true" ]; then
    echo "Press 'q' in the video window to quit"
else
    echo "Running in headless mode (console output only)"
fi
echo ""

eval $CMD

