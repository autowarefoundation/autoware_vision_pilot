# **CarlaControlPublisher 节点**
# **CarlaControlPublisher Node**

## **概述**
## **Overview**

`CarlaControlPublisher` 节点是一个 ROS 2 接口，用于将转向和纵向（油门和制动）控制命令合并到 **CARLA 仿真器**。它订阅转向和油门命令话题，并向 CARLA 自车发布相应的控制消息。
The `CarlaControlPublisher` node is a ROS 2 interface for merging steering and longitudinal (throttle & brake) control commands to the **CARLA Simulator**. It subscribes to steering and throttle command topics and publishes corresponding control messages to the CARLA ego vehicle.

## **发布的话题**
## **Published Topics**

| 话题 / Topic | 消息类型 / Message Type | 描述 / Description |
|--------|---------------|-------------|
| `/carla/hero/vehicle_control_cmd` | `carla_msgs/msg/CarlaEgoVehicleControl` | 发布组合的油门、转向和制动命令，以控制 CARLA 中的自车 / Publishes combined throttle, steering, and braking commands to control the ego vehicle in CARLA. |

## **订阅的话题**
## **Subscribed Topics**

| 话题 / Topic | 消息类型 / Message Type | 描述 / Description |
|--------|---------------|-------------|
| `/vehicle/steering_cmd` | `std_msgs/msg/Float32` | 接收期望轮胎转向角（弧度）/ Receives desired tire steering angle in radians |
| `/vehicle/throttle_cmd` | `std_msgs/msg/Float32` | 接收归一化油门命令输入（-1.0 到 1.0）。负值表示通过制动减速 / Receives normalized throttle command input (-1.0 to 1.0). Negative is to reduce speed by braking |

## **参数**
## **Parameters**

| 名称 / Name | 类型 / Type | 默认值 / Default | 描述 / Description |
|------|------|----------|-------------|
| `publish_rate` | float | 10.0 Hz | 向 CARLA 发布控制消息的频率 / Frequency at which control messages are published to CARLA. |

## **使用示例**
## **Example Usage**

### **运行节点**
### **Run the Node**

```bash
ros2 run carla_control_publisher pub_carla_control 
```
