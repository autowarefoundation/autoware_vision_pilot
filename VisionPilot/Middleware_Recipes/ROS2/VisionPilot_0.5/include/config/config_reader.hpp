/**
 * @file config_reader.hpp
 * @brief Configuration file reader for EgoLanes pipeline (.conf format)
 */

#pragma once

#include <string>
#include <map>

namespace autoware_pov::config {

struct Config {
    bool simulation;
    double velocity;
    std::string mode;
    struct {
        std::string video_path;
        bool camera_auto_select;
        std::string camera_device_id;
    } source;
    struct {
        std::string egolanes_path;
        std::string provider;
        std::string precision;
        int device_id;
        std::string cache_dir;
        float threshold;
        std::string autosteer_path;
    } models;
    struct {
        double Kp;
        double Ki;
        double Kd;
        double Ks;
    } steering_control;
    struct {
        bool enable_viz;
        bool save_video;
        std::string output_video_path;
        bool measure_latency;
        std::string csv_log_path;
    } output;
    struct {
        bool enabled;
        bool spawn_viewer;
        std::string save_path;
    } rerun;
    struct {
        bool enabled;
        std::string interface_name;
    } can_interface;
};

class ConfigReader {
public:
    static Config loadFromFile(const std::string& config_path);
    
private:
    static std::map<std::string, std::string> parseConfigFile(const std::string& config_path);
    static std::string trim(const std::string& str);
    static bool parseBool(const std::string& value);
    static int parseInt(const std::string& value);
    static double parseDouble(const std::string& value);
    static float parseFloat(const std::string& value);
};

} // namespace autoware_pov::config
