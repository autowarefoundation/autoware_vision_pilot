#ifndef VISIONPILOT_RADAR_ROS2_INTERFACE_HPP
#define VISIONPILOT_RADAR_ROS2_INTERFACE_HPP

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class RadarRos2Interface
{
public:
    struct Return {
        float range_m = 0.f;
        float azimuth_rad = 0.f;
        float range_rate = 0.f;
    };

    explicit RadarRos2Interface(std::string topic);
    ~RadarRos2Interface();

    // Closest buffered scan to camera_stamp_ns. Empty if none within slop_ms.
    std::vector<Return> take_closest(int64_t camera_stamp_ns, int slop_ms);

private:
    class Node : public rclcpp::Node
    {
    public:
        Node(const std::string& topic, RadarRos2Interface* parent);
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    };

    void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    std::shared_ptr<Node> node_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;

    struct Scan {
        int64_t stamp_ns = 0;
        std::vector<Return> points;
    };
    std::mutex mutex_;
    std::deque<Scan> buf_;
};

#endif
