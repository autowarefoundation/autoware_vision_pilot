# VisionPilot ⇄ CARLA (0.9.16) — ROS 2

要将 VisionPilot 连接到 CARLA 进行闭环仿真，且不引入 carla_bridge 依赖，需要两个节点：一个发布车速，另一个发布命令消息以便 CARLA 驱动车辆。
In order to connect VisionPilot to CARLA for closed loop simulation and to not introduce carla_bridge dependency, two nodes are required, one to publish vehicle speed and one to publish a command message so CARLA can drive the vehicle.

`carla_vehicle_speed_publisher` 包将车速发布到 `/vehicle/speed` 话题，`carla_control_publisher` 将控制消息发布到 `/carla/hero/vehicle_control_cmd`。
`carla_vehicle_speed_publisher` package publishes the vehicle speed to `/vehicle/speed` topic, and `carla_control_publisher` publish control messages to `/carla/hero/vehicle_control_cmd`.
