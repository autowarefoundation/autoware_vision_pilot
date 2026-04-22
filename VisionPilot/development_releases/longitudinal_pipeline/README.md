# VisionPilot 0.9 – Lateral + Longitudinal Release

## Description
Vision Pilot 0.9 processes images from a single front-facing camera to enable both ADAS features and highway autopilot within a single driving lane. Compared to the prior release, Vision Pilot 0.5, this version of the system incorporates both lateral perception and planning modules alongside longitudinal perception and planning modules in two parallel streams, through the additional integration of the AutoSpeed neural network for closest-in-path-object detection. This enables features such as Autonomous Cruise Control, Forward Collision Warning, and Automatic Emergency braking. In order to estimate the distance of the closest-in-path-object, a homography transform is utilized which maps image pixels to road coordinates, providing an estimate of real-world distances in metres. A Kalman filter is used to track the distance of the closest-in-path-object and estimate its velocity. To maintain a safe following distance to the lead vehicle, the system complies with [Mobileye's Responsibility Sensitive Safety framework](https://www.mobileye.com/technology/responsibility-sensitive-safety/).

**System Architecture**

<img src="../../Media/VisionPilot_0.9.png" width="100%">

Installation Ubuntu 22.04 X86 System

## 1. Install ONNX Runtime

**Skip this step if you have  already installed onnxruntime-linux-x64-gpu-1.22.0.tgz**

```bash
cd Downloads

wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz
tar -xvzf onnxruntime-linux-x64-gpu-1.22.0.tgz
cd onnxruntime-linux-x64-gpu-1.22.0
cp -r onnxruntime-linux-x64-gpu-1.22.0 $HOME/ #or any other folder you like to have

export ONNXRUNTIME_ROOT=/home/YourUser/onnxruntime-linux-x64-gpu-1.22.0 #if you add cp it to another folder, change it to this folder
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH

permanent
echo 'export ONNXRUNTIME_ROOT=/home/YourUser/onnxruntime-linux-x64-gpu-1.22.0' >> ~/.bashrc
source ~/.bashrc
```

## 2. Install TensorRT 

### Check if TensorRT is installed
```bash
dpkg -l | grep tensorrt
or
dpkg -l | grep nvinfer
```

### If you don't see a TensorRT version printed, then install TensorRT

**Visit the download page and download the correct package for your Nvidia GPU:**
https://developer.nvidia.com/tensorrt

**Once it is downloaded, enter the download folder:**
```bash
cd ~/Downloads
```

**Install TensorRT**
```bash
# adapt to your downloaded version of TensorRT
tar -xvzf TensorRT-10.x.x.Linux.x86_64-gnu.cuda-12.x.tar.gz 
```

**Place in correct folder and export TensorRT**
```bash
sudo mv TensorRT-10.0.1.6 /opt/tensorrt

export TENSORRT_ROOT=/opt/tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# or permanent

echo 'export TENSORRT_ROOT=/opt/tensorrt' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cd /opt/tensorrt/python
pip install tensorrt-*.whl
```

**Test to ensure TensorRT has been installed correctly**
```bash
python # or python3
import tensorrt as trt
print(trt.__version__)
```

## 3. Build

From `Production_Releases/0.9`:

```bash
mkdir -p build
cd build
cmake ..      # ONNX Runtime + TensorRT (uses $ONNXRUNTIME_ROOT)
make -j$(nproc)
cd ..
```

Ensure:
- `ONNXRUNTIME_ROOT` points to your ONNX Runtime GPU install.
- TensorRT/CUDA are installed.

## 4. Download the AI models

**Create directories where the AI models will be stored**
```bash
mkdir -p autoware_projects/weights
cd autoware_projects/weights
mkdir AutoSpeed
mkdir Autosteer
mkdir EgoLanes
```

**Download and copy the ONNX models to corresponding folders**

AutoSpeed: https://drive.google.com/file/d/1Zhe8uXPbrPr8cvcwHkl1Hv0877HHbxbB/view?usp=drive_link

AutoSteer: Please contact admin at zain.khawaja@autoware.org

EgoLanes: https://drive.google.com/file/d/1b4jAoH6363ggTgVU0b6URbFfcOL3-r1Q/view?usp=sharing


## 5. Configure (`visionpilot.conf`)

Edit `visionpilot.conf` in this directory:

- **Mode & source**
  - `mode=video` or `mode=camera`
  - `source.video.path=/path/to/video.mp4`
- **Camera calibration (optional)**
  - `camera_calibration.enabled=true/false`
  - `camera_calibration.inference_camera_config_path=<INFERENCE_CAM_CONFIG_PATH>`
  - `camera_calibration.standard_pose_config_path=<STANDARD_POSE_CONFIG_PATH>`
  - Uses JSON schema from `VisionPilot/calibration/configs/`
- **Models**
  - `models.egolanes.path=.../Egolanes_fp32.onnx`
  - `models.autosteer.path=.../AutoSteer_FP32.onnx`
  - `models.autospeed.path=.../AutoSpeed_n.onnx`
  - `models.homography_yaml.path=.../homography_2.yaml`
- **Timing**
  - `pipeline.target_fps=10.0`
- **Lateral PID**
  - `steering_control.Kp/Ki/Kd/Ks`
- **Longitudinal**
  - `longitudinal.autospeed.conf_thresh`
  - `longitudinal.autospeed.iou_thresh`
  - `longitudinal.ego_speed_default_ms` (used when CAN is disabled/invalid)
  - `longitudinal.pid.Kp/Ki/Kd`
- **CAN**
  - `can_interface.enabled=true/false`
  - `can_interface.interface_name=can0`

## 6. Run

```bash
./run_final.sh           # uses /usr/share/visionpilot/visionpilot.conf if present
./run_final.sh ./visionpilot.conf   # explicit config path
```

You should see:
- EgoLanes + AutoSteer lateral pipeline initialization
- AutoSpeed + ObjectFinder longitudinal initialization
- “Lateral and Longitudinal pipelines running in PARALLEL…”

## 7. Shared Memory Outputs

The process publishes a single shared-memory segment with all outputs:

- Name: `/visionpilot_state`
- Struct: `VisionPilotState` (see `include/publisher/visionpilot_shared_state.hpp`)
  - Lateral: steering angles, PathFinder CTE/yaw/curvature, lane departure flag
  - Longitudinal: CIPO distance/velocity, RSS safe distance, ideal speed, FCW/AEB flags, longitudinal control effort
  - CAN/ego: speed, steering angle, validity

### Quick test reader

From `0.9`:

```bash
./tools/shm_reader          # live view while visionpilot is running
./tools/shm_reader --once   # single snapshot
```

If VisionPilot is running correctly you will see frame IDs increasing and CIPO / steering values updating. When VisionPilot stops, `shm_reader` will show the last published frame until the segment is unlinked.
