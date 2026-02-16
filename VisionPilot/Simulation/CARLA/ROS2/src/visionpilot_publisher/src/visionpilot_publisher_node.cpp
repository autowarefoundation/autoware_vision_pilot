#include <cstdio>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "visionpilot_publisher_node.hpp"


VisionPilotPublisher::VisionPilotPublisher() : Node("visionpilot_publisher_node"), ipcSharedState("/ctrl_shm")
{
  speed_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/speed", 1.0);
  brake_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/brake_cmd", 1.0);

  TARGET_VEL = 23.6;  // 80 km/h in m/s, actual 25m/s after adding steady state error

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&VisionPilotPublisher::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "✅ VisionPilot publisher node initialized and running");
}

void VisionPilotPublisher::timer_callback()
{
  auto state = ipcSharedState.get();
  if (state) {
    // std::cout << "[SharedState] Steering: " << state->steering << ", Velocity: " << state->velocity << std::endl;
    auto throttle_msg = std_msgs::msg::Float32();
    throttle_msg.data = 0.0;

    throttle_msg.data = state->velocity;
    speed_pub_->publish(throttle_msg);
  }
}


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VisionPilotPublisher>());
  rclcpp::shutdown();
  return 0;
}
