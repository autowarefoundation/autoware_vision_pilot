/**
 * @file config_reader.cpp
 * @brief Configuration file reader implementation (.conf format)
 */

#include "config/config_reader.hpp"
#include <fstream>
#include <algorithm>
#include <cctype>

namespace autoware_pov::config {

Config ConfigReader::loadFromFile(const std::string& config_path) {
    auto props = parseConfigFile(config_path);
    Config config;

    config.simulation = parseBool(props["simulation"]);
    config.velocity = parseDouble(props["velocity"]);

    config.mode = props["mode"];
    
    if (config.mode == "video") {
        config.source.video_path = props["source.video.path"];
    } else {
        config.source.camera_auto_select = parseBool(props["source.camera.auto_select"]);
        config.source.camera_device_id = props["source.camera.device_id"];
    }
    
    config.models.egolanes_path = props["models.egolanes.path"];
    config.models.provider = props["models.egolanes.provider"];
    config.models.precision = props["models.egolanes.precision"];
    config.models.device_id = parseInt(props["models.egolanes.device_id"]);
    config.models.cache_dir = props["models.egolanes.cache_dir"];
    config.models.threshold = parseFloat(props["models.egolanes.threshold"]);
    config.models.autosteer_path = props["models.autosteer.path"];
    
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
    
    return config;
}

std::map<std::string, std::string> ConfigReader::parseConfigFile(const std::string& config_path) {
    std::ifstream file(config_path);
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

