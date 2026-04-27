/**
 * @file main.cpp
 * @brief Multi-threaded AutoSpeed 2.0 Longitudinal Control Pipeline
 * 
 * Architecture:
 * - Capture Thread: Reads frames from video source or camera
 * - AutoSpeed Inference Thread: Runs AutoSpeed 2.0 model for object detection
 * - Speed Planning Thread: Computes safe speed targets (ACC/FCW/AEB)
 * - Longitudinal Control Thread: PID controller for speed tracking
 * - Publisher Thread: Shares results via IPC
 * Basically a modular pipeline cloned from VisionPilot 1.0 but with only
 * longitudinal control and AutoSpeed 2.0 model, for specific testing.
 */

#include "inference/autospeed/onnxruntime_engine.hpp"
#include "visualization/visualize_long.hpp"
#include "tracking/object_finder.hpp"
#include "speed_planning/speed_planning.hpp"
#include "longitudinal/pi_controller.hpp"
#include "publisher/visionpilot_shared_state.hpp"
#include "config/config_reader.hpp"
#include <thread>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef VISIONPILOT_SHARE_DIR
#define VISIONPILOT_SHARE_DIR "."
#endif

using namespace autoware_pov::vision::autospeed;
using namespace autoware_pov::vision::tracking;
using namespace autoware_pov::vision::speed_planning;
using namespace autoware_pov::vision::longitudinal;
using namespace autoware_pov::vision::publisher;
using namespace autoware_pov::config;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    try {
        std::cout << "\n========================================\n"
                  << "VisionPilot AutoSpeed 2.0 Longitudinal Control\n"
                  << "========================================\n" << std::endl;

        // Load configuration
        ConfigReader config;
        std::cout << "[Main] Configuration loaded\n";
        
        // Initialize AutoSpeed 2.0 inference engine (ONNX Runtime, CPU by default)
        std::cout << "[Main] Initializing AutoSpeed 2.0 inference engine...\n";
        const std::string model_path = std::string(VISIONPILOT_SHARE_DIR) + "/weights/autospeed_2.onnx";
        
        AutoSpeedOnnxEngine autospeed_engine(
            model_path,
            "cpu",        // provider: cpu or tensorrt
            "fp32",       // precision: fp32 or fp16 (TensorRT only)
            0,            // device_id for GPU
            "./trt_cache" // cache directory for TensorRT
        );
        
        std::cout << "[Main] AutoSpeed 2.0 model loaded from: " << model_path << "\n"
                  << "[Main] Initialization complete\n" << std::endl;

        // Default homography calibration (identity matrix for testing)
        // In production, load from actual calibration file
        const std::string homography_yaml = "config/homography.yaml";
        
        // Initialize object tracker
        std::cout << "[Main] Initializing object tracking...\n";
        ObjectFinder object_finder(
            homography_yaml,  // homography calibration file
            1920,             // image width
            1280              // image height
        );
        
        // Initialize speed planner with initial state
        std::cout << "[Main] Initializing speed planning module...\n";
        SpeedPlanner speed_planner(
            0.0,    // initial relative_cipo_speed (m/s)
            100.0,  // initial cipo_distance (m)
            0.0,    // initial ego_speed (m/s)
            0.0,    // initial absolute_cipo_speed (m/s)
            false   // is_cipo_present
        );
        
        // Initialize longitudinal controller (PI with reasonable gains)
        std::cout << "[Main] Initializing PI controller...\n";
        PIController pi_controller(
            0.5,   // K_p: Proportional gain
            0.1,   // K_i: Integral gain
            0.05   // K_d: Derivative gain
        );
        
        // Initialize IPC publisher
        std::cout << "[Main] Initializing IPC publisher...\n";
        VisionPilotSharedState publisher("/visionpilot_state", true);
        std::cout << "[Main] IPC publisher initialized\n\n";

        // Main processing loop
        std::cout << "[Main] AutoSpeed 2.0 pipeline initialized successfully.\n"
                  << "[Main] Waiting for camera input...\n\n";

        // Open video capture
        cv::VideoCapture cap;
        bool has_camera = false;
        
        // Try to open camera first
        cap.open(0);
        if (cap.isOpened()) {
            has_camera = true;
            std::cout << "[Main] Camera opened successfully\n";
        } else {
            std::cerr << "[Main] No camera detected. Running in idle mode.\n"
                      << "[Main] To test with video file, modify main.cpp\n\n";
        }

        bool running = true;
        int frame_count = 0;
        double current_ego_speed = 0.0;  // m/s
        
        auto start_time = high_resolution_clock::now();

        while (running && frame_count < 5000) {  // Process up to 5000 frames
            cv::Mat frame;
            
            if (has_camera && cap.read(frame)) {
                frame_count++;
                
                // Resize frame if needed
                if (frame.cols > 1920 || frame.rows > 1080) {
                    cv::resize(frame, frame, cv::Size(1920, 1080));
                }

                // Run AutoSpeed 2.0 inference
                std::vector<Detection> detections = autospeed_engine.inference(frame);
                
                // Track detected objects
                std::vector<TrackedObject> tracked_objects = object_finder.update(detections, frame);
                
                // Find CIPO (Closest In-Path Object) for speed planning
                double cipo_distance = 100.0;  // Default: no object
                double cipo_velocity = 0.0;
                bool cipo_exists = false;
                
                if (!tracked_objects.empty()) {
                    // Find closest object
                    const TrackedObject& closest = tracked_objects[0];
                    cipo_distance = closest.distance_m;
                    cipo_velocity = closest.velocity_ms;
                    cipo_exists = true;
                }
                
                // Update speed planner with current state
                speed_planner.setIsCIPOPresent(cipo_exists);
                speed_planner.setEgoSpeed(current_ego_speed);
                if (cipo_exists) {
                    speed_planner.setCIPOState(cipo_velocity, cipo_distance);
                }
                
                // Compute ideal speed
                double ideal_speed = speed_planner.calcIdealDrivingSpeed();
                
                // Compute control signal using PI controller
                double control_effort = pi_controller.computeEffort(current_ego_speed, ideal_speed);
                
                // Prepare shared state
                VisionPilotState state;
                state.seq = 0;
                state.frame_number = frame_count;
                
                // Longitudinal outputs
                state.cipo_exists = cipo_exists;
                state.cipo_track_id = cipo_exists ? tracked_objects[0].track_id : -1;
                state.cipo_class_id = cipo_exists ? tracked_objects[0].class_id : 0;
                state.cipo_distance_m = cipo_distance;
                state.cipo_velocity_ms = cipo_velocity;
                state.ideal_speed_ms = ideal_speed;
                state.safe_distance_m = speed_planner.calcSafeRSSDistance();
                state.fcw_active = speed_planner.getFCWState();
                state.aeb_active = speed_planner.getAEBState();
                state.control_effort_ms2 = control_effort;
                state.ego_speed_ms = current_ego_speed;
                state.can_valid = false;  // No CAN data in this demo
                
                // Publish state
                publisher.publish(state);
                
                // Log progress every 30 frames
                if (frame_count % 30 == 0) {
                    std::cout << "[Frame " << std::setw(4) << frame_count << "] "
                              << "CIPO: " << (cipo_exists ? "YES" : "NO") << " | "
                              << "Distance: " << std::fixed << std::setprecision(2) 
                              << cipo_distance << "m | "
                              << "Target Speed: " << ideal_speed << " m/s | "
                              << "Tracked Objects: " << tracked_objects.size() << "\n";
                }
            } else {
                // No camera or end of video
                if (!has_camera) {
                    // Idle mode - publish placeholder state periodically
                    if (frame_count == 0) {
                        VisionPilotState idle_state;
                        idle_state.seq = 0;
                        idle_state.frame_number = 0;
                        idle_state.cipo_exists = false;
                        idle_state.can_valid = false;
                        publisher.publish(idle_state);
                        
                        std::cout << "[Main] Idle state published. Ready for IPC clients.\n";
                        std::cout << "[Main] Press Ctrl+C to exit.\n\n";
                    }
                }
                std::this_thread::sleep_for(milliseconds(100));
                frame_count++;
                
                if (frame_count > 100) break;  // Exit idle after 100 iterations
            }
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(end_time - start_time);

        std::cout << "\n[Main] AutoSpeed 2.0 pipeline shutting down...\n"
                  << "[Main] Processed " << frame_count << " frames in " 
                  << duration.count() << " seconds\n"
                  << "[Main] Goodbye!\n" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Main] Error: " << e.what() << std::endl;
        return 1;
    }
}
