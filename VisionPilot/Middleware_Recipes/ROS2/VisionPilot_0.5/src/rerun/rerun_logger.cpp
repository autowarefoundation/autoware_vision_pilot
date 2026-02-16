/**
 * @file rerun_logger.cpp
 * @brief Implementation of Rerun logger for inference visualization
 */

#include "rerun/rerun_logger.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace autoware_pov::vision::rerun_integration {

RerunLogger::RerunLogger(const std::string& app_id, bool spawn_viewer, const std::string& save_path)
    : enabled_(false)
{
#ifdef ENABLE_RERUN
    // CRITICAL: Don't create RecordingStream if there's no output sink!
    // If no viewer and no save path, don't initialize at all to prevent memory buffering
    if (!spawn_viewer && save_path.empty()) {
        std::cout << "ℹ Rerun not initialized (no viewer or save path specified)" << std::endl;
        return;
    }
    
    try {
        rec_ = std::make_unique<rerun::RecordingStream>(app_id);
        
        // Save to file FIRST (before spawn) to ensure connection
        if (!save_path.empty()) {
            auto result = rec_->save(save_path);
            if (result.is_err()) {
                std::cerr << "Failed to save to " << save_path << std::endl;
                if (!spawn_viewer) {
                    return;
                }
            } else {
                std::cout << "✓ Saving to: " << save_path << std::endl;
            }
        }
        
        // Spawn viewer (streams data directly - no RAM buffering!)
        if (spawn_viewer) {
            rerun::SpawnOptions opts;
            opts.memory_limit = "2GB";
            
            auto result = rec_->spawn(opts);
            if (result.is_err()) {
                std::cerr << "Failed to spawn Rerun viewer" << std::endl;
                if (save_path.empty()) {
                    return;
                }
            } else {
                std::cout << "✓ Rerun viewer spawned (memory limit: 2GB)" << std::endl;
            }
        }
        
        enabled_ = true;
        std::cout << "✓ Rerun logging enabled (all frames, deep clone mode)" << std::endl;
        if (!spawn_viewer && !save_path.empty()) {
            std::cout << "  ⚠ WARNING: Save-only mode buffers ALL data in RAM until completion!" << std::endl;
            std::cout << "    Recommended: Use spawn viewer for real-time streaming (no buffering)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Rerun initialization failed: " << e.what() << std::endl;
    }
#else
    (void)app_id;
    (void)spawn_viewer;
    (void)save_path;
    std::cout << "ℹ Rerun support not compiled in (use -DENABLE_RERUN=ON)" << std::endl;
#endif
}

RerunLogger::~RerunLogger() = default;

void RerunLogger::logData(
    int frame_number,
    const cv::Mat& resized_frame,
    const autoware_pov::vision::egolanes::LaneSegmentation& lanes,
    const cv::Mat& stacked_view,
    const autoware_pov::drivers::CanVehicleState& vehicle_state,
    double steering_angle_raw,
    double steering_angle,
    float autosteer_angle,
    const autoware_pov::vision::path_planning::PathFinderOutput& path_output,
    long inference_time_us)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Set timeline
    rec_->set_time_sequence("frame", frame_number);
    
    // Log resized input frame (convert BGR→RGB using fixed buffer, then borrow)
    cv::cvtColor(resized_frame, rgb_buffer_, cv::COLOR_BGR2RGB);
    logImage("camera/image", rgb_buffer_);
    
    // Log lane masks (borrow directly - data is still in scope)
    logMask("lanes/ego_left", lanes.ego_left);
    logMask("lanes/ego_right", lanes.ego_right);
    logMask("lanes/other", lanes.other_lanes);
    
    // Log visualization (convert BGR→RGB, then borrow)
    cv::cvtColor(stacked_view, rgb_buffer_, cv::COLOR_BGR2RGB);
    logImage("visualization/stacked_view", rgb_buffer_);
    
    // Log CAN bus data (scalars)
    rec_->log("can/steering_angle_deg", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({
                  rerun::components::Scalar(vehicle_state.is_valid ? vehicle_state.steering_angle_deg : 0.0)
              })));
    rec_->log("can/speed_kmph", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({
                  rerun::components::Scalar(vehicle_state.is_valid ? vehicle_state.speed_kmph : 0.0)
              })));
    
    // Log control outputs (all in degrees)
    rec_->log("control/pid_steering_raw_deg", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(steering_angle_raw)})));
    rec_->log("control/pid_steering_filtered_deg", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(steering_angle)})));
    rec_->log("control/autosteer_angle_deg", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(autosteer_angle)})));
    
    // Log PathFinder outputs (scalars)
    if (path_output.fused_valid) {
        rec_->log("pathfinder/cte", 
                  rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(path_output.cte)})));
        rec_->log("pathfinder/yaw_error", 
                  rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(path_output.yaw_error)})));
        rec_->log("pathfinder/curvature", 
                  rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(path_output.curvature)})));
    }
    
    // Log inference time metric
    double time_ms = inference_time_us / 1000.0;
    rec_->log("metrics/inference_time_ms", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>({rerun::components::Scalar(time_ms)})));
#else
    (void)frame_number;
    (void)resized_frame;
    (void)lanes;
    (void)stacked_view;
    (void)vehicle_state;
    (void)steering_angle_raw;
    (void)steering_angle;
    (void)autosteer_angle;
    (void)path_output;
    (void)inference_time_us;
#endif
}

void RerunLogger::logImage(const std::string& entity_path, const cv::Mat& image)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // ZERO-COPY: Borrow data directly from cv::Mat buffer (fixed buffer, reused every frame)
    // Safe because:
    // 1. Caller cloned/downsampled the data before calling us
    // 2. We use fixed rgb_buffer_ that persists across frames
    // 3. rerun::borrow() creates non-owning view
    // 4. Rerun serializes synchronously before we return
    size_t data_size = image.cols * image.rows * image.channels();
    
    rec_->log(
        entity_path,
        rerun::Image(
            rerun::borrow(image.data, data_size),  
            rerun::WidthHeight(
                static_cast<uint32_t>(image.cols),
                static_cast<uint32_t>(image.rows)
            ),
            rerun::ColorModel::RGB  // Already converted BGR→RGB in logData
        )
    );
#else
    (void)entity_path;
    (void)image;
#endif
}

void RerunLogger::logMask(const std::string& entity_path, const cv::Mat& mask)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Convert float mask to uint8 using fixed buffer (reused every mask)
    // This is the only allocation we keep - unavoidable for type conversion
    mask.convertTo(mask_u8_buffer_, CV_8UC1, 255.0);
    
    // ZERO-COPY: Borrow data directly from fixed buffer
    // Safe because:
    // 1. mask_u8_buffer_ is a member variable that persists
    // 2. rerun::borrow() creates non-owning view
    // 3. Rerun serializes synchronously before we return
    // 4. Buffer is reused for next mask (no allocation churn)
    size_t data_size = mask_u8_buffer_.cols * mask_u8_buffer_.rows;
    
    rec_->log(
        entity_path,
        rerun::archetypes::DepthImage(
            rerun::borrow(mask_u8_buffer_.data, data_size),  // ✅ Zero-copy borrow!
            rerun::WidthHeight(
                static_cast<uint32_t>(mask_u8_buffer_.cols),
                static_cast<uint32_t>(mask_u8_buffer_.rows)
            ),
            rerun::datatypes::ChannelDatatype::U8
        )
    );
#else
    (void)entity_path;
    (void)mask;
#endif
}

} // namespace autoware_pov::vision::rerun_integration

