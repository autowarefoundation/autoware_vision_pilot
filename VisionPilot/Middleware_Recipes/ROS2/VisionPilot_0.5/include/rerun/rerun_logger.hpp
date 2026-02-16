/**
 * @file rerun_logger.hpp
 * @brief Minimal Rerun logger for inference visualization
 */

#pragma once

#ifdef ENABLE_RERUN
#include <rerun.hpp>
#endif

#include "inference/lane_segmentation.hpp"
#include "drivers/can_interface.hpp"
#include "path_planning/path_finder.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace autoware_pov::vision::rerun_integration {

/**
 * @brief Rerun logger for complete frame data
 * 
 * Logs:
 * - Resized input frame (320x640)
 * - Lane segmentation masks (filtered)
 * - Final visualization (stacked view)
 * - CAN bus data (steering, speed)
 * - Control outputs (PID, AutoSteer)
 * - PathFinder outputs (CTE, yaw, curvature)
 * - Inference time metrics
 */
class RerunLogger {
public:
    /**
     * @brief Initialize Rerun recording stream
     * @param app_id Application identifier
     * @param spawn_viewer If true, spawn viewer; if false, save to file
     * @param save_path Path to .rrd file (used if !spawn_viewer)
     */
    RerunLogger(const std::string& app_id = "EgoLanes", 
                bool spawn_viewer = true,
                const std::string& save_path = "");
    
    ~RerunLogger();
    
    /**
     * @brief Log all frame data (frame, CAN bus, control outputs, visualization)
     * @param frame_number Frame sequence number
     * @param resized_frame Resized input frame (320x640, BGR)
     * @param lanes Lane segmentation masks
     * @param stacked_view Final visualization (BGR)
     * @param vehicle_state CAN bus data (steering, speed)
     * @param steering_angle_raw Raw PID output before filtering (degrees)
     * @param steering_angle Filtered PID output (final steering, degrees)
     * @param autosteer_angle AutoSteer steering angle (degrees)
     * @param path_output PathFinder output (CTE, yaw, curvature)
     * @param inference_time_us Inference time in microseconds
     */
    void logData(
        int frame_number,
        const cv::Mat& resized_frame,
        const autoware_pov::vision::egolanes::LaneSegmentation& lanes,
        const cv::Mat& stacked_view,
        const autoware_pov::drivers::CanVehicleState& vehicle_state,
        double steering_angle_raw,
        double steering_angle,
        float autosteer_angle,
        const autoware_pov::vision::path_planning::PathFinderOutput& path_output,
        long inference_time_us);
    
    /**
     * @brief Check if Rerun is enabled and initialized
     */
    bool isEnabled() const { return enabled_; }

private:
#ifdef ENABLE_RERUN
    std::unique_ptr<rerun::RecordingStream> rec_;
    
    // Fixed buffers for zero-allocation logging (reused every frame)
    cv::Mat rgb_buffer_;        // Buffer for BGR→RGB conversion
    cv::Mat mask_u8_buffer_;    // Buffer for float→uint8 mask conversion
#endif
    bool enabled_;
    
    void logImage(const std::string& entity_path, const cv::Mat& image);
    void logMask(const std::string& entity_path, const cv::Mat& mask);
};

} // namespace autoware_pov::vision::rerun_integration

