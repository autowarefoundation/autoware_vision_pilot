# **CarlaControlPublisher 节点**

[🇺🇸 English](README.md)

## **概述**

`CarlaControlPublisher` 节点是一个 ROS 2 接口，用于将转向和纵向（油门和制动）控制命令合并到 **CARLA 仿真器**。它订阅转向和油门命令话题，并向 CARLA 自车发布相应的控制消息。

## **发布的话题**

| 话题 | 消息类型 | 描述 |
|--------|---------------|-------------|
| `/carla/hero/vehicle_control_cmd` | `carla_msgs/msg/CarlaEgoVehicleControl` | 发布组合的油门、转向和制动命令，以控制 CARLA 中的自车 |

## **订阅的话题**

| 话题 | 消息类型 | 描述 |
|--------|---------------|-------------|
| `/vehicle/steering_cmd` | `std_msgs/msg/Float32` | 接收期望轮胎转向角（弧度） |
| `/vehicle/throttle_cmd` | `std_msgs/msg/Float32` | 接收归一化油门命令输入（-1.0 到 1.0）。负值表示通过制动减速 |

## **参数**

| 名称 | 类型 | 默认值 | 描述 |
|------|------|----------|-------------|
| `publish_rate` | float | 10.0 Hz | 向 CARLA 发布控制消息的频率 |

## **使用示例**

### **运行节点**

```bash
ros2 run carla_control_publisher pub_carla_control 
```
