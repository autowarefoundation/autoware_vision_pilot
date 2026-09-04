cmake_minimum_required(VERSION 3.22.1)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(radar_ros2_interface
        src/radar_ros2_interface.cpp
)

target_include_directories(radar_ros2_interface
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(radar_ros2_interface
        PUBLIC
        rclcpp::rclcpp
        sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)

target_compile_features(radar_ros2_interface PUBLIC cxx_std_17)
