/**
 * @file config_reader.cpp
 * @brief Configuration file reader implementation (.conf format)
 */

#include "config/config_reader.hpp"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace autoware_pov::config {

Config ConfigReader::loadFromFile(const std::string& config_path) {
    auto props = parseConfigFile(config_path);
    Config config;
    
    config.mode = props["mode"];
    
    if (config.mode == "video") {
        config.source.video_path = props["source.video.path"];
    } else {
        config.source.camera_auto_select = parseBool(props["source.camera.auto_select"]);
        config.source.camera_device_id = props["source.camera.device_id"];
    }

    config.models.egolanes_path =
        props.find("models.egolanes.path") != props.end()
            ? props["models.egolanes.path"]
            : "";
    config.models.provider =
        props.find("models.egolanes.provider") != props.end()
            ? props["models.egolanes.provider"]
            : "cpu";
    config.models.precision =
        props.find("models.egolanes.precision") != props.end()
            ? props["models.egolanes.precision"]
            : "fp32";
    config.models.device_id =
        props.find("models.egolanes.device_id") != props.end() && !props["models.egolanes.device_id"].empty()
            ? parseInt(props["models.egolanes.device_id"])
            : 0;
    config.models.cache_dir =
        props.find("models.egolanes.cache_dir") != props.end()
            ? props["models.egolanes.cache_dir"]
            : "./trt_cache";
    config.models.threshold =
        props.find("models.egolanes.threshold") != props.end() && !props["models.egolanes.threshold"].empty()
            ? parseFloat(props["models.egolanes.threshold"])
            : 0.0f;
    config.models.autosteer_path =
        props.find("models.autosteer.path") != props.end()
            ? props["models.autosteer.path"]
            : "";
    config.models.autospeed_path =
        props.find("models.autospeed.path") != props.end()
            ? props["models.autospeed.path"]
            : "";
    config.models.homography_yaml_path =
        props.find("models.homography_yaml.path") != props.end()
            ? props["models.homography_yaml.path"]
            : "";
    
    config.steering_control.Kp = parseDouble(props["steering_control.Kp"]);
    config.steering_control.Ki = parseDouble(props["steering_control.Ki"]);
    config.steering_control.Kd = parseDouble(props["steering_control.Kd"]);
    config.steering_control.Ks = parseDouble(props["steering_control.Ks"]);
    
    config.output.enable_viz = parseBool(props["output.enable_viz"]);
    config.output.save_video = parseBool(props["output.save_video"]);
    config.output.output_video_path = props["output.output_video_path"];
    config.output.measure_latency = parseBool(props["output.measure_latency"]);
    config.output.csv_log_path = props["output.csv_log_path"];
    
    // config.rerun.enabled = parseBool(props["rerun.enabled"]);
    // config.rerun.spawn_viewer = parseBool(props["rerun.spawn_viewer"]);
    // config.rerun.save_path = props["rerun.save_path"];
    config.rerun.enabled = props.find("rerun.enabled") != props.end() ? parseBool(props["rerun.enabled"]) : false;
    config.rerun.spawn_viewer = props.find("rerun.spawn_viewer") != props.end() ? parseBool(props["rerun.spawn_viewer"]) : true;
    config.rerun.save_path = props.find("rerun.save_path") != props.end() ? props["rerun.save_path"] : "";
    
    config.can_interface.enabled = parseBool(props["can_interface.enabled"]);
    config.can_interface.interface_name = props["can_interface.interface_name"];

    config.camera_calibration.enabled =
        props.find("camera_calibration.enabled") != props.end()
            ? parseBool(props["camera_calibration.enabled"])
            : true;
    config.camera_calibration.inference_camera_config_path =
        props.find("camera_calibration.inference_camera_config_path") != props.end()
            ? props["camera_calibration.inference_camera_config_path"]
            : "";
    config.camera_calibration.standard_pose_config_path =
        props.find("camera_calibration.standard_pose_config_path") != props.end()
            ? props["camera_calibration.standard_pose_config_path"]
            : "";
 
    // Longitudinal & pipeline tuning (with sensible defaults if keys are missing)
    config.longitudinal.autospeed_conf_thresh =
        props.find("longitudinal.autospeed.conf_thresh") != props.end()
            ? parseFloat(props["longitudinal.autospeed.conf_thresh"])
            : 0.5f;
    config.longitudinal.autospeed_iou_thresh =
        props.find("longitudinal.autospeed.iou_thresh") != props.end()
            ? parseFloat(props["longitudinal.autospeed.iou_thresh"])
            : 0.5f;
    config.longitudinal.ego_speed_default_ms =
        props.find("longitudinal.ego_speed_default_ms") != props.end()
            ? parseDouble(props["longitudinal.ego_speed_default_ms"])
            : 10.0;
    config.longitudinal.pid_Kp =
        props.find("longitudinal.pid.Kp") != props.end()
            ? parseDouble(props["longitudinal.pid.Kp"])
            : 0.5;
    config.longitudinal.pid_Ki =
        props.find("longitudinal.pid.Ki") != props.end()
            ? parseDouble(props["longitudinal.pid.Ki"])
            : 0.1;
    config.longitudinal.pid_Kd =
        props.find("longitudinal.pid.Kd") != props.end()
            ? parseDouble(props["longitudinal.pid.Kd"])
            : 0.05;

    config.capture_fps =
        props.find("pipeline.target_fps") != props.end()
            ? parseDouble(props["pipeline.target_fps"])
            : 10.0;

    return config;
}

std::map<std::string, std::string> ConfigReader::parseConfigFile(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    std::map<std::string, std::string> props;
    std::string line;
    
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) eq_pos = line.find(':');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        
        if (!value.empty() && ((value[0] == '"' && value.back() == '"') || 
                                (value[0] == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.length() - 2);
        }
        
        if (!key.empty()) props[key] = value;
    }
    
    return props;
}

std::string ConfigReader::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

bool ConfigReader::parseBool(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
}

int ConfigReader::parseInt(const std::string& value) {
    return std::stoi(value);
}

double ConfigReader::parseDouble(const std::string& value) {
    return std::stod(value);
}

float ConfigReader::parseFloat(const std::string& value) {
    return std::stof(value);
}

} // namespace autoware_pov::config

