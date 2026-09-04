#include <radar_ros2_interface/radar_ros2_interface.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

int64_t stamp_ns(const builtin_interfaces::msg::Time& t)
{
    return static_cast<int64_t>(t.sec) * 1000000000LL + static_cast<int64_t>(t.nanosec);
}

bool has_field(const sensor_msgs::msg::PointCloud2& msg, const char* name)
{
    for (const auto& f : msg.fields)
        if (f.name == name) return true;
    return false;
}

std::vector<RadarRos2Interface::Return>
parse_cloud(const sensor_msgs::msg::PointCloud2& msg)
{
    std::vector<RadarRos2Interface::Return> out;
    if (msg.data.empty()) return out;

    const char* rr_name = nullptr;
    if (has_field(msg, "range_rate")) rr_name = "range_rate";
    else if (has_field(msg, "doppler")) rr_name = "doppler";
    else if (has_field(msg, "velocity")) rr_name = "velocity";

    auto push = [&](float range, float az, float rr) {
        if (range <= 0.f) return;
        out.push_back({range, az, rr});
    };

    if (has_field(msg, "x") && has_field(msg, "y")) {
        sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y");
        if (rr_name) {
            sensor_msgs::PointCloud2ConstIterator<float> rr(msg, rr_name);
            for (; x != x.end(); ++x, ++y, ++rr)
                push(std::hypot(*x, *y), std::atan2(*y, *x), *rr);
        } else {
            for (; x != x.end(); ++x, ++y)
                push(std::hypot(*x, *y), std::atan2(*y, *x), 0.f);
        }
        return out;
    }
    if (has_field(msg, "range") && has_field(msg, "azimuth")) {
        sensor_msgs::PointCloud2ConstIterator<float> r(msg, "range"), a(msg, "azimuth");
        if (rr_name) {
            sensor_msgs::PointCloud2ConstIterator<float> rr(msg, rr_name);
            for (; r != r.end(); ++r, ++a, ++rr)
                push(*r, *a, *rr);
        } else {
            for (; r != r.end(); ++r, ++a)
                push(*r, *a, 0.f);
        }
    }
    return out;
}

}  // namespace

RadarRos2Interface::Node::Node(const std::string& topic, RadarRos2Interface* parent)
    : rclcpp::Node("RadarRos2Node")
{
    auto qos = rclcpp::QoS(rclcpp::KeepLast(8)).reliable().durability_volatile();
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        topic, qos,
        [parent](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            parent->on_cloud(msg);
        });
    RCLCPP_INFO(get_logger(), "RadarRos2Interface  topic=%s", topic.c_str());
}

RadarRos2Interface::RadarRos2Interface(std::string topic)
{
    if (!rclcpp::ok())
    {
        static int argc = 1;
        static const char* argv[] = {"radar_ros2_interface", nullptr};
        rclcpp::init(argc, const_cast<char**>(argv));
    }
    node_ = std::make_shared<Node>(topic, this);
    executor_.add_node(node_);
    spin_thread_ = std::thread([this]() { executor_.spin(); });
}

RadarRos2Interface::~RadarRos2Interface()
{
    executor_.cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
}

void RadarRos2Interface::on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (!msg) return;
    Scan scan;
    scan.stamp_ns = stamp_ns(msg->header.stamp);
    scan.points = parse_cloud(*msg);
    std::lock_guard<std::mutex> lock(mutex_);
    buf_.push_back(std::move(scan));
    while (buf_.size() > 8) buf_.pop_front();
}

std::vector<RadarRos2Interface::Return>
RadarRos2Interface::take_closest(int64_t camera_stamp_ns, int slop_ms)
{
    const int64_t slop_ns = static_cast<int64_t>(std::max(0, slop_ms)) * 1000000LL;
    std::lock_guard<std::mutex> lock(mutex_);
    if (buf_.empty() || camera_stamp_ns == 0)
        return {};

    int64_t best_dt = slop_ns + 1;
    const Scan* best = nullptr;
    for (const auto& s : buf_) {
        const int64_t dt = std::llabs(s.stamp_ns - camera_stamp_ns);
        if (dt < best_dt) { best_dt = dt; best = &s; }
    }
    if (best && best_dt <= slop_ns)
        return best->points;
    return {};
}
