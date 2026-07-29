# Vision Pilot — 开源 L2 级 ADAS

[🇺🇸 English Version](README.md)

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)">
        <img src="./Media/VisionPilot_logo.png" alt="VisionPilot" width="100%">
    </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/953808765935816715?label=Autoware%20Discord)](https://discord.com/invite/Q94UsPvReQ)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/autowarefoundation/autoware.privately-owned-vehicles)
![GitHub Repo stars](https://img.shields.io/github/stars/autowarefoundation/autoware.privately-owned-vehicles)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=whit)
![ROS](https://img.shields.io/badge/ROS-22314E?style=for-the-badge&logo=ROS&logoColor=whit)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/the-autoware-foundation)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@autowarefoundation)
[![Website](https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://autoware.org/)
</div>

<div align="center">

⭐ 在 GitHub 上给我们 Star — 你的支持是我们前进的动力！

</div>

## 免费且完全开源的 L2 级 ADAS 方案

[![观看视频](/Media/Vision_Pilot_Thumbnail.jpg)](https://drive.google.com/file/d/1pAxpppljBdDKFYgrKdWiwUtPiEBYbVhL/view)

**本代码库包含一个可量产且可安全认证的开源 L2 级 ADAS 系统实现，名为 Vision Pilot。**

Vision Pilot 旨在与汽车 OEM 和一级供应商的量产乘用车集成，该系统也可选装于公交车和卡车的运输与物流场景。

我们免费提供完整的代码库（包括 AI 模型权重），以帮助普及这项关键技术。Vision Pilot 采用宽松的 Apache 2.0 许可证，可用于商业和研究目的。

<img src="/Media/VisionPilot_1.0.png" width="100%">

Vision Pilot 旨在支持车道内自动驾驶的基础/入门级 L2 ADAS 功能，包括以下功能：

- **ACC** — 自适应巡航控制
- **FCW** — 前方碰撞预警
- **AEB** — 自动紧急制动
- **LKAS** — 车道保持辅助
- **LDW** — 车道偏离预警
- **ISA** — 智能限速辅助
- **Autopilot** — 单车道高速公路免手扶自动驾驶

**传感器规格：**

Vision Pilot 只需一个前置单目摄像头即可运行，水平视场角 52-55 度，分辨率 1MP-2MP。

### 混合端到端 AI 架构

Vision Pilot 的核心采用**混合端到端 AI 架构**，数据由感知 AI 模型（保障安全）和端到端 AI 模型（提升性能）并行处理。

<img src="/Media/VisionPilot_architecture.png" width="100%">

具体而言，Vision Pilot 利用了 Autoware 基金会开发的三个开源 AI 模型：

1. [AutoSpeed](https://github.com/autowarefoundation/auto_speed) — 最近同路径目标检测
2. [AutoSteer](https://github.com/autowarefoundation/auto_steer) — 自车路径未来航路点检测
3. [AutoDrive](https://github.com/autowarefoundation/auto_drive) — 端到端距离/同路径目标存在检测和道路曲率估计

### 不依赖高精地图

**Vision Pilot 不需要 3D 高精地图**，以"无地图"模式实时跟随道路。

### 介绍演示

如需了解更多关于 Vision Pilot 的信息，请查看我们的介绍演示：[链接](https://canva.link/qjz6rwp40v7apft)

## 快速开始

有三种方式可以构建和运行 Vision Pilot — 我们提供示例数据供可视化和评估使用，请从以下选项中选择：

<details>
<summary><b>选项一 — 下载并从源码构建</b></summary>

要开始使用本项目，请从以下地址下载源码：

```bash
  git clone https://github.com/autowarefoundation/autoware_vision_pilot.git
```

从 GitHub [releases](https://github.com/microsoft/onnxruntime/releases) 页面下载 ONNX Runtime。

构建项目：

```bash
  cd VisionPilot
```

```bash
  mkdir build && cd build
```

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> ../
```

或启用 ROS2 支持：

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> -DENABLE_ROS2_INTERFACE=ON ../
```

```bash
  make
```

这将构建项目并在 build 目录中生成 VisionPilot 可执行文件。

#### 用测试数据运行 Vision Pilot 并可视化输出

**OpenLane 数据集：**

要使用开环场景测试 Vision Pilot，请先从 [Google Drive](https://drive.google.com/drive/folders/1-Sxgz3XHzFD6XtETz1sVFRtDKY3W57QB?usp=sharing) 目录下载示例数据。

该目录包含由数据集中序列图像数据组成的视频，以及从数据集中提取的车速数据。

更新 `config` 目录中的 VisionPilot 配置文件 `vision_pilot.conf`，设置：

```
source.mode             = video
```

并在 `vision_pilot_test.conf` 中设置：

```
source.input_video         = <INPUT_VIDEO_FILE_PATH>
source.input_vehicle_speed = <INPUT_VEHICLE_SPEED_FILE_PATH>
```

指向对应的视频文件路径和车速文件路径。

*注意*：从源码构建 VisionPilot 时，请在构建前更新配置文件。

从 `build` 目录运行 VisionPilot：

```
./VisionPilot
```

#### 构建 VisionPilot DEB 包

要构建 VisionPilot 的 Debian DEB 包，运行：

```bash
  cpack -G DEB
```

要构建仅支持 CPU 的 VisionPilot DEB 包，使用以下配置构建：

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> -DGPU=OFF ../
```

```bash
  make
```

```bash
  cpack -G DEB
```

</details>

<details>
<summary><b>选项二 — 使用预编译 DEB 包</b></summary>

### 从预编译 DEB 包安装

此方法推荐用于新系统安装且尚未安装 CUDA 依赖的情况。

下载 [VisionPilot](https://github.com/autowarefoundation/autoware_vision_pilot.git) 预编译二进制。

安装 .deb 包：

```bash
  sudo apt install ./VisionPilot-1.0-x86_64.deb
```

重启系统后，VisionPilot 的 CUDA 依赖将自动安装。

#### 用测试数据运行 Vision Pilot 并可视化输出

**OpenLane 数据集：**

要使用开环场景测试 Vision Pilot，请先从 [Google Drive](https://drive.google.com/drive/folders/1-Sxgz3XHzFD6XtETz1sVFRtDKY3W57QB?usp=sharing) 目录下载示例数据。

该目录包含由数据集中序列图像数据组成的视频，以及从数据集中提取的车速数据。

更新 VisionPilot 配置文件 `vision_pilot.conf`，设置：

```
source.mode             = video
```

并在 `vision_pilot_test.conf` 中设置：

```
source.input_video         = <INPUT_VIDEO_FILE_PATH>
source.input_vehicle_speed = <INPUT_VEHICLE_SPEED_FILE_PATH>
```

指向对应的视频文件路径和车速文件路径。

*注意*：从预编译二进制安装 VisionPilot 时，配置文件位于：

```
/usr/share/visionpilot/config
```

目录。

从命令行运行 VisionPilot：

```
VisionPilot
```

</details>

<details>
<summary><b>选项三 — 使用 Docker 容器</b></summary>

### 在 Docker 容器中运行

要在 Docker 容器中运行 Vision Pilot，请使用仓库中 docker 目录提供的 Dockerfile 构建容器。

Docker 容器可构建为支持 GPU/CPU，以及 NO_ROS2/ROS2 支持。

要构建容器，进入 docker 子目录并运行以下命令：

默认为 GPU 支持，无 ROS2 支持：

```bash
  ./build.sh --gpu --ros2 
```

构建 CPU 支持版本：

```bash
  ./build.sh --cpu
```

使用 `run.sh` 脚本运行容器。例如以 CPU 支持运行：

```bash
  ./run.sh --cpu
```

或以 GPU 支持和 ROS2 支持运行：

```bash
  ./run.sh --gpu --ros2
```

*注意*：构建容器前，请更新 `config` 目录中的配置文件。

如果构建 CPU 支持版本，更新 `config/vision_pilot.conf`：

```
engine.provider     = cpu
```

构建 ROS2 支持版本时更新：

```
source.mode         = ros2
```

使用视频输入源时更新 `config/vision_pilot_test.conf`：

```
source.input_video         = <INPUT_VIDEO_FILE_PATH>
source.input_vehicle_speed = <INPUT_VEHICLE_SPEED_FILE_PATH>
```

可使用 `run.sh` 脚本的 `--data` 开关修改输入目录：

```bash
  ./run.sh --gpu --data <HOST_DIR>:<CONTAINER_DIR>
```

*注意*：`<CONTAINER_DIR>` 需要与容器构建时保持一致。

</details>

## 在模拟器中运行 Vision Pilot

我们支持 CARLA 模拟器 0.9.16 版本，用于在虚拟环境中对 Vision Pilot 进行闭环测试。

CARLA 0.9.16 与 Unreal Engine 4：https://carla.readthedocs.io/en/latest/

### 如何安装 CARLA 模拟器

1. 按照官方文档下载二进制和依赖：https://carla-ue5.readthedocs.io/en/latest/start_quickstart/#

2. 如果你的 GPU 显存 ≤ 6GB，请参考以下修改以在较低显存下运行 CARLA（已在 RTX3060 笔记本版上测试）：https://gist.github.com/xmfcx/a5e32fdecfcd85c6cc9d472ce7a3a98d

### 如何配合 Vision Pilot 运行 CARLA 模拟器

将以下文件路径修改为 CARLA 的下载和运行位置。确保将 `--volume` 路径修改为你本地的具体目录路径。

```sh
  docker run -it --rm \
  --runtime=nvidia \                        # 使用 NVIDIA 运行时获取 GPU 访问权限
  --net=host \                              # 使用主机网络栈（有助于网络/性能）
  --env=DISPLAY=$DISPLAY \                  # 传递主机的 DISPLAY 环境变量（用于 GUI 转发）
  --env=NVIDIA_VISIBLE_DEVICES=all \        # 向容器暴露所有 GPU
  --env=NVIDIA_DRIVER_CAPABILITIES=all \    # 启用所有驱动能力（图形、计算等）
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \ # 挂载 X11 UNIX socket 以启用 GUI 应用显示
  --volume="$HOME/Downloads/carla/CARLA_0.9.16/:/home/carla/host-carla" \ 
                                            # 按需修改：将本地 CARLA 文件夹挂载到容器中
  --workdir="/home/carla/host-carla" \      # 设置工作目录为挂载的 CARLA 文件夹
  carlasim/carla:0.9.16 \                   # 使用官方 CARLA Docker 镜像，版本 0.9.16
  bash CarlaUE4.sh -nosound                 # 运行 CARLA 启动脚本，带 -nosound 参数
```

要使用 ROS2 原生接口运行，在末尾添加 `--ros2`：

#### 更新 VisionPilot 配置

更新 `vision_pilot.conf`，设置 `source.mode`：

```bash
 source.mode = ros2 
```

同时更新 `vision_pilot_ros2.conf`，设置 `source.input_camera_topic`：

```bash
  source.input_camera_topic = /carla/hero/main_cam/image
```

#### 构建 VisionPilot

构建带 ROS2 支持的 VisionPilot：

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> -DENABLE_ROS2_INTERFACE=ON ../
```

#### 构建 CARLA 桥接

进入 Simulation/CARLA/ROS2 目录并构建 CARLA ROS2 桥接：

```bash
  colcon build
```

#### 运行 CARLA 桥接

加载安装目录：

```bash
  source ./install/setup.bash
```

```bash
  ros2 launch  carla_bridge_bringup carla_bridge.launch.py host:=<HOST> port:=<PORT>
```

*注意*：如果你在不同于运行 Vision Pilot 的机器上运行 CARLA，需要指定 host 和 port 参数。将运行 CARLA 的机器 IP 地址设为 host，将 CARLA 发布的端口设为 port。

#### 运行 VisionPilot

进入 VisionPilot build 目录并运行 VisionPilot：

```bash
  ./VisionPilot
```

[![观看视频](/Media/Vision_Pilot_CARLA.jpg)](https://drive.google.com/file/d/1DCtXkKnhGTcU-YRiBCTTbCYkixUw8FZW/view?usp=sharing)

## 使用你自己的相机运行 Vision Pilot

要使用你自己的相机运行 Vision Pilot，你需要校准相机并将校准信息提供给 Vision Pilot 应用 — 这对 Vision Pilot 准确测量道路形状和目标距离非常重要。

### 校准你的相机

请按照[校准指南](/Calibration/)中的步骤操作，将校准 yaml 文件保存在 [Vision Pilot 配置文件夹](/VisionPilot/config/) 中，并用你的校准文件参数覆盖 [H.yaml](/VisionPilot/config/H.yaml) 文件的数据 — 建议保留一份原始 H.yaml 参数的副本，以便你能在我们的示例数据上运行 Vision Pilot。

## 路线图

- 支持前视摄像头与车载雷达的融合
- 支持 8MP 分辨率、120 度水平视场角
- 安全验证与汽车标准合规（ISO26262, ISO8800）

## 参与贡献

要了解如何参与本项目，请阅读[上手指南](/ONBOARDING.md)。
