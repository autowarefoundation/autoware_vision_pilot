# Zenoh

[🇺🇸 English](README.md)

本目录提供基于 Zenoh 的 CARLA 桥接及相关流水线。

## 演示

- CARLA + Zenoh：Vision 流水线演示

[![CARLA + Zenoh Vision Pipelines Demo](https://img.youtube.com/vi/TQ6JwexlXvo/hqdefault.jpg)](https://youtu.be/TQ6JwexlXvo "CARLA + Zenoh: Vision Pipelines Demo")

## 依赖

请参阅以下页面中描述的依赖：

[VisionPilot/Middleware_Recipes/Zenoh](../../../../VisionPilot/Middleware_Recipes/Zenoh)

## 使用方法

### 设置（运行一次）

```sh
just setup
```

### 构建

```sh
# 构建全部
export LIBTORCH_INSTALL_ROOT=/path/to/libtorch/
export ONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-gpu-1.22.0
just build
# 可选（分别构建各组件）
just build_bridge
just build_video_pubsub
just build_models
```

### 运行

#### 启动 CARLA 服务器

```sh
just run_carla
```

启动 CARLA 仿真器。

Docker 镜像首次下载可能需要较长时间。

#### 启动 Zenoh CARLA 桥接

```sh
just run_zenoh
```

启动带 pygame 控制的 CARLA Python 代理，同时启动 Zenoh CARLA 桥接。

#### 运行流水线

```sh
# 原始相机视图
just run_carla_sub

# SceneSeg
just run_carla_sceneseg

# DomainSeg
just run_carla_domainseg

# Scene3D
just run_carla_scene3d

# Egolanes
just run_carla_egolanes
```

### 清理

```sh
just clean
```
