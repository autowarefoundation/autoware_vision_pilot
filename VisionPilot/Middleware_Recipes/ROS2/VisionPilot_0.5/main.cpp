/**
 * @file main.cpp
 * @brief Multi-threaded EgoLanes lane detection inference pipeline
 * 
 * Architecture:
 * - Capture Thread: Reads frames from video source or camera
 * - Inference Thread: Runs lane detection model
 * - Display Thread: Optionally visualizes and saves results
 */

#ifdef SKIP_ORT
// Use TensorRT directly
#include "inference/tensorrt_engine.hpp"
#include "inference/tensorrt_autosteer_engine.hpp"
using EgoLanesEngine =
  autoware_pov::vision::egolanes::EgoLanesTensorRTEngine;
using AutoSteerEngine =
  autoware_pov::vision::egolanes::AutoSteerTensorRTEngine;
#else
// Use ONNX Runtime
#include "inference/onnxruntime_engine.hpp"
#include "inference/autosteer_engine.hpp"
using EgoLanesEngine =
  autoware_pov::vision::egolanes::EgoLanesOnnxEngine;
using AutoSteerEngine =
  autoware_pov::vision::egolanes::AutoSteerOnnxEngine;
#endif
#include "visualization/draw_lanes.hpp"
#include "lane_filtering/lane_filter.hpp"
#include "lane_tracking/lane_tracking.hpp"
#include "camera/camera_utils.hpp"
#include "path_planning/path_finder.hpp"
#include "steering_control/steering_controller.hpp"
#include "steering_control/steering_filter.hpp"
#include "drivers/can_interface.hpp"
#include "config/config_reader.hpp"
#ifdef ENABLE_RERUN
#include "rerun/rerun_logger.hpp"
#endif
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <boost/circular_buffer.hpp>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
 #include <fstream>
 #include <cmath>
 #include <limits>
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif

#ifndef VISIONPILOT_SHARE_DIR
#define VISIONPILOT_SHARE_DIR "."
#endif

#include "publisher/ipc_shared_state.hpp"

using namespace autoware_pov::vision::egolanes;
using namespace autoware_pov::vision::camera;
using namespace autoware_pov::vision::path_planning;
using namespace autoware_pov::vision::steering_control;
using namespace autoware_pov::drivers;
using namespace autoware_pov::config;
using namespace std::chrono;
using namespace visionpilot::publisher;

Config config;

IpcSharedState ipc("/ctrl_shm");

// Thread-safe queue with max size limit
template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait if queue is full (backpressure)
        cond_not_full_.wait(lock, [this] {
            return queue_.size() < max_size_ || !active_;
        });
        if (!active_) return;

        queue_.push(item);
        cond_not_empty_.notify_one();
    }

    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = queue_.front();
        queue_.pop();
        cond_not_full_.notify_one();  // Notify that space is available
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_empty_.wait(lock, [this] { return !queue_.empty() || !active_; });
        if (!active_ && queue_.empty()) {
            return T();
        }
        T item = queue_.front();
        queue_.pop();
        cond_not_full_.notify_one();  // Notify that space is available
        return item;
    }

    void stop() {
        active_ = false;
        cond_not_empty_.notify_all();
        cond_not_full_.notify_all();
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    std::atomic<bool> active_{true};
    size_t max_size_;
};

// Timestamped frame
struct TimestampedFrame {
    cv::Mat frame;
    int frame_number;
    steady_clock::time_point timestamp;
    CanVehicleState vehicle_state; // Ground truth from CAN
};

// Inference result
struct InferenceResult {
    cv::Mat frame;
    cv::Mat resized_frame_320x640;  // Resized frame for Rerun logging (320x640)
    LaneSegmentation lanes;
    DualViewMetrics metrics;
    int frame_number;
    steady_clock::time_point capture_time;
    steady_clock::time_point inference_time;
    double steering_angle_raw = 0.0;  // Raw PID output before filtering (degrees)
    double steering_angle = 0.0;  // Filtered PID output (final steering, degrees)
    PathFinderOutput path_output; // Added for metric debug
    float autosteer_angle = 0.0f;  // Steering angle from AutoSteer (degrees)
    bool autosteer_valid = false;  // Whether AutoSteer ran (skips first frame)
    CanVehicleState vehicle_state; // CAN bus data from capture thread
};

// Performance metrics
struct PerformanceMetrics {
    std::atomic<long> total_capture_us{0};
    std::atomic<long> total_inference_us{0};
    std::atomic<long> total_display_us{0};
    std::atomic<long> total_end_to_end_us{0};
    std::atomic<int> frame_count{0};
    bool measure_latency{true};
};

/**
 * @brief Transform BEV pixel coordinates to BEV metric coordinates (meters)
 *
 * Transformation based on 640x640 BEV image:
 * Input (Pixels):
 *   - Origin (0,0) at Top-Left
 *   - x right, y down
 *   - Vehicle at Bottom-Center (320, 640)
 *
 * Output (Meters):
 *   - Origin (0,0) at Vehicle Position
 *   - x right (lateral), y forward (longitudinal)
 *   - Range: X [-20m, 20m], Y [0m, 40m]
 *   - Scale: 640 pixels = 40 meters
 *
 * @param bev_pixels BEV points in pixel coordinates (from LaneTracker)
 * @return BEV points in metric coordinates (meters, x=lateral, y=longitudinal)
 */
std::vector<cv::Point2f> transformPixelsToMeters(const std::vector<cv::Point2f>& bev_pixels) {
    std::vector<cv::Point2f> bev_meters;

    if (bev_pixels.empty()) {
        return bev_meters;
    }


    const double bev_width_px = 640.0;
    const double bev_height_px = 640.0;
    const double bev_range_m = 40.0;

    const double scale = bev_range_m / bev_height_px; // 40m / 640px = 0.0625 m/px
    const double center_x = bev_width_px / 2.0;       // 320.0
    const double origin_y = bev_height_px;            // 640.0 (bottom)
    //check again
    for (const auto& pt : bev_pixels) {
        bev_meters.push_back(cv::Point2f(
            (pt.x - center_x) * scale,       // Lateral: (x - 320) * scale
            (origin_y - pt.y) * scale       // Longitudinal: (640 - y) * scale (Flip Y to match image origin)
        ));
    }

    return bev_meters;
}

cv::VideoCapture openGStreamer(const std::string& pipeline) {
  cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  return cap;
}

/**
 * @brief Unified capture thread - handles both video files and cameras
 */
void captureThread(
    const std::string& source,
    bool is_camera,
    ThreadSafeQueue<TimestampedFrame>& queue,
    PerformanceMetrics& metrics,
    std::atomic<bool>& running,
    CanInterface* can_interface = nullptr)
{
    cv::VideoCapture cap;

    if (config.simulation) {
      std::cout << "Opening video stream: " << source << std::endl;
      std::string pipeline = "udpsrc port=5000 caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                              "rtph264depay ! decodebin ! videoconvert ! appsink";
      cap = openGStreamer(pipeline);
    } else if (is_camera) {
         std::cout << "Opening camera: " << source << std::endl;
         cap = openCamera(source);
     } else {
         std::cout << "Opening video: " << source << std::endl;
         cap.open(source);
     }

    if (!cap.isOpened()) {
         std::cerr << "Failed to open source: " << source << std::endl;
        running.store(false);
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
     double fps = cap.get(cv::CAP_PROP_FPS);

     std::cout << "Source opened: " << frame_width << "x" << frame_height
               << " @ " << fps << " FPS\n" << std::endl;

     if (!is_camera) {
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << total_frames << std::endl;
     }

     // For camera: throttle 30fps → 10fps
     int frame_skip = 0;
     int skip_interval = is_camera ? 3 : 1;

    int frame_number = 0;
    while (running.load()) {
        auto t_start = steady_clock::now();

        cv::Mat frame;
         if (!cap.read(frame) || frame.empty()) {
             if (is_camera) {
                 std::cerr << "Camera error" << std::endl;
             } else {
            std::cout << "End of video stream" << std::endl;
             }
            break;
        }

         auto t_end = steady_clock::now();

         // Frame throttling
         if (++frame_skip < skip_interval) continue;
         frame_skip = 0;

        long capture_us = duration_cast<microseconds>(t_end - t_start).count();
        metrics.total_capture_us.fetch_add(capture_us);

        // Poll CAN interface if available
        CanVehicleState current_state;
        if (can_interface) {
            can_interface->update(); // Poll socket
            current_state = can_interface->getState();
            std::cout << "CAN State: " << current_state.steering_angle_deg << std::endl;
        }

        TimestampedFrame tf;
        tf.frame = frame;
        tf.frame_number = frame_number++;
        tf.timestamp = t_end;
        tf.vehicle_state = current_state;
        queue.push(tf);
    }

    running.store(false);
    queue.stop();
     cap.release();
}

/**
 * @brief Inference thread - runs lane detection model
 */
void inferenceThread(
    EgoLanesEngine& engine,
    ThreadSafeQueue<TimestampedFrame>& input_queue,
    ThreadSafeQueue<InferenceResult>& output_queue,
    PerformanceMetrics& metrics,
    std::atomic<bool>& running,
     float threshold,
     PathFinder* path_finder = nullptr,
     SteeringController* steering_controller = nullptr,
     AutoSteerEngine* autosteer_engine = nullptr
)
{
    // Init lane filter
    LaneFilter lane_filter(0.5f);

     // Init lane tracker
     LaneTracker lane_tracker;

     // Init SteeringFilter
     SteeringFilter steering_filter(0.05f);
     // AutoSteer: Circular buffer for raw EgoLanes tensors [1, 3, 80, 160]
     // Stores copies of last 2 frames for temporal inference
     const int EGOLANES_TENSOR_SIZE = 3 * 80 * 160;  // 38,400 floats per frame
     boost::circular_buffer<std::vector<float>> egolanes_tensor_buffer(2);

     // Pre-allocated concatenation buffer for AutoSteer input [1, 6, 80, 160]
     std::vector<float> autosteer_input_buffer(6 * 80 * 160);  // 76,800 floats


    while (running.load()) {
        TimestampedFrame tf = input_queue.pop();
        if (tf.frame.empty()) continue;

        auto t_inference_start = steady_clock::now();
        // std::cout<< "Frame size before cropping: " << tf.frame.size() << std::endl;
         // Crop tf.frame 420 pixels top
         // tf.frame = tf.frame(cv::Rect(
         //     0,
         //     420,
         //     tf.frame.cols,
         //     tf.frame.rows - 420
         // ));
        //  std::cout<< "Frame size: " << tf.frame.size() << std::endl;

        // Run Ego Lanes inference
        LaneSegmentation raw_lanes = engine.inference(tf.frame, threshold);

        // ========================================
        // AUTOSTEER INTEGRATION
        // ========================================
        float autosteer_steering = 0.0f;
        double steering_angle_raw = 0.0;
        double steering_angle = 0.0;

        // 1. Copy raw EgoLanes tensor [1, 3, 80, 160] for temporal buffer
        const float* raw_tensor = engine.getRawTensorData();
        std::vector<float> current_tensor(EGOLANES_TENSOR_SIZE);
        std::memcpy(current_tensor.data(), raw_tensor, EGOLANES_TENSOR_SIZE * sizeof(float));

        // 2. Store in circular buffer (auto-deletes oldest when full)
        egolanes_tensor_buffer.push_back(std::move(current_tensor));

        // 3. Run AutoSteer only when buffer is full (skip first frame)
        if (egolanes_tensor_buffer.full()) {
            // Concatenate t-1 and t into pre-allocated buffer
            std::memcpy(autosteer_input_buffer.data(),
                       egolanes_tensor_buffer[0].data(),  // t-1
                       EGOLANES_TENSOR_SIZE * sizeof(float));

            std::memcpy(autosteer_input_buffer.data() + EGOLANES_TENSOR_SIZE,
                       egolanes_tensor_buffer[1].data(),  // t
                       EGOLANES_TENSOR_SIZE * sizeof(float));

            // Run AutoSteer inference
            autosteer_steering = autosteer_engine->inference(autosteer_input_buffer);
        }
        // ========================================
        // Post-processing with lane filter
        LaneSegmentation filtered_lanes = lane_filter.update(raw_lanes);

         // Further processing with lane tracker
         cv::Size frame_size(tf.frame.cols, tf.frame.rows);
         std::pair<LaneSegmentation, DualViewMetrics> track_result = lane_tracker.update(
             filtered_lanes,
             frame_size
         );

         LaneSegmentation final_lanes = track_result.first;
         DualViewMetrics final_metrics = track_result.second;

        auto t_inference_end = steady_clock::now();

        // Calculate inference latency
        long inference_us = duration_cast<microseconds>(
            t_inference_end - t_inference_start).count();
        metrics.total_inference_us.fetch_add(inference_us);

          // ========================================
          // PATHFINDER (Polynomial Fitting + Bayes Filter) + STEERING CONTROL
          // ========================================
          PathFinderOutput path_output; // Declaring at higher scope for result storage
          path_output.fused_valid = false; // Initialize as invalid

          if (final_metrics.bev_visuals.valid) {

              // 1. Get BEV points in PIXEL space from LaneTracker
              std::vector<cv::Point2f> left_bev_pixels = final_metrics.bev_visuals.bev_left_pts;
              std::vector<cv::Point2f> right_bev_pixels = final_metrics.bev_visuals.bev_right_pts;

              // 2. Transform BEV pixels → BEV meters
              // TODO: Calibrate transformPixelsToMeters() for your specific camera
              std::vector<cv::Point2f> left_bev_meters = transformPixelsToMeters(left_bev_pixels);
              std::vector<cv::Point2f> right_bev_meters = transformPixelsToMeters(right_bev_pixels);

              // 3. Update PathFinder (polynomial fit + Bayes filter in metric space)
              // Pass AutoSteer steering angle (replaces computed curvature)
              path_output = path_finder->update(left_bev_meters, right_bev_meters, autosteer_steering);

              // 4. Compute steering angle
              if (path_output.fused_valid) {
                  steering_angle_raw = steering_controller->computeSteering(
                      path_output.cte,
                      path_output.yaw_error * 180 / M_PI,
                      path_output.curvature
                  );
              }

              // Filter the raw PID output
              steering_angle = steering_filter.filter(steering_angle_raw, 0.1);

              // 5. Print output (cross-track error, yaw error, curvature, lane width + variances + steering)
              if (path_output.fused_valid) {
                  std::cout << "[Frame " << tf.frame_number << "] "
                            << "CTE: " << std::fixed << std::setprecision(3) << path_output.cte << " m "
                            << "(var: " << path_output.cte_variance << "), "
                            << "Yaw: " << path_output.yaw_error << " rad "
                            << "(var: " << path_output.yaw_variance << "), "
                            << "Curv: " << path_output.curvature << " 1/m "
                            << "(var: " << path_output.curv_variance << "), "
                            << "Width: " << path_output.lane_width << " m "
                            << "(var: " << path_output.lane_width_variance << ")";

                  // PID Steering output
                  double pid_deg = steering_angle;
                  std::cout << " | PID: " << std::setprecision(2) << pid_deg << " deg";

                  // AutoSteer output (if valid)
                  if (egolanes_tensor_buffer.full()) {
                      std::cout << " | AutoSteer: " << std::setprecision(2) << autosteer_steering << " deg";

                      // Show difference
                      double diff = autosteer_steering - pid_deg;
                      std::cout << " (Δ: " << std::setprecision(2) << diff << " deg)";
                  }

                  std::cout << std::endl;
              } else if (egolanes_tensor_buffer.full()) {
                  // If PathFinder is not valid but AutoSteer is running, still log AutoSteer
                  std::cout << "[Frame " << tf.frame_number << "] "
                            << "AutoSteer: " << std::fixed << std::setprecision(2) << autosteer_steering << " deg "
                            << "(PathFinder: invalid)" << std::endl;
              }
          }
          // ========================================

        // Package result (clone frame to avoid race with capture thread reusing buffer)
        InferenceResult result;
        // std::cout<< "Frame size: " << tf.frame.size() << std::endl;
        result.frame = tf.frame.clone();  // Clone for display thread safety

        // Resize frame to 640x320 for Rerun logging (only if Rerun enabled, but prepare anyway)
        cv::resize(tf.frame, result.resized_frame_320x640, cv::Size(640, 320), 0, 0, cv::INTER_AREA);
        result.lanes = final_lanes;
        result.metrics = final_metrics;
        result.frame_number = tf.frame_number;
        result.capture_time = tf.timestamp;
        result.inference_time = t_inference_end;
        result.steering_angle_raw = steering_angle_raw;  // Store raw PID output (before filtering)
        result.steering_angle = steering_angle;  // Store filtered PID output (final steering)
        result.path_output = path_output;        // Store for viz
        result.autosteer_angle = autosteer_steering;  // Store AutoSteer angle
        result.autosteer_valid = egolanes_tensor_buffer.full();
        result.vehicle_state = tf.vehicle_state;  // Copy CAN bus data
        output_queue.push(result);

        // publish steering angle and constant velocioty
        ipc.set(steering_angle, config.velocity);
    }

    output_queue.stop();
}

/**
 * @brief Display thread - handles visualization and video saving
 */
void displayThread(
    ThreadSafeQueue<InferenceResult>& queue,
    PerformanceMetrics& metrics,
    std::atomic<bool>& running,
    bool enable_viz,
    bool save_video,
     const std::string& output_video_path,
     const std::string& csv_log_path
#ifdef ENABLE_RERUN
    , autoware_pov::vision::rerun_integration::RerunLogger* rerun_logger = nullptr
#endif
)
{
    // Visualization setup
     int window_width = 640;
     int window_height = 320;
    if (enable_viz) {
         cv::namedWindow(
             "EgoLanes Inference",
             cv::WINDOW_NORMAL
         );
         cv::resizeWindow(
             "EgoLanes Inference",
             window_width,
             window_height
         );
    }

    // Video writer setup
    cv::VideoWriter video_writer;
    bool video_writer_initialized = false;

    if (save_video && enable_viz) {
        std::cout << "Video saving enabled. Output: " << output_video_path << std::endl;
    }

     // CSV logger for all steering outputs (PathFinder + PID + AutoSteer)
     std::ofstream csv_file;
     csv_file.open(csv_log_path);
     if (csv_file.is_open()) {
        // Write header
        csv_file << "frame_id,timestamp_ms,"
                 << "orig_lane_offset,orig_yaw_offset,orig_curvature,"
                 << "pathfinder_cte,pathfinder_yaw_error,pathfinder_curvature,"
                 << "pid_steering_raw_deg,pid_steering_filtered_deg,"
                 << "autosteer_angle_deg,autosteer_valid\n";

         std::cout << "CSV logging enabled: " << csv_log_path << std::endl;
     } else {
         std::cerr << "Error: could not open " << csv_log_path << " for writing" << std::endl;
    }

    // Load steering wheel images
  std::string predSteeringImagePath = std::string(VISIONPILOT_SHARE_DIR) + "/images/wheel_green.png";
  cv::Mat predSteeringWheelImg = cv::imread(predSteeringImagePath, cv::IMREAD_UNCHANGED);
  std::string gtSteeringImagePath = std::string(VISIONPILOT_SHARE_DIR) + "/images/wheel_white.png";
  cv::Mat gtSteeringWheelImg = cv::imread(gtSteeringImagePath, cv::IMREAD_UNCHANGED);

    while (running.load()) {
        InferenceResult result;
        if (!queue.try_pop(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto t_display_start = steady_clock::now();

        int count = metrics.frame_count.fetch_add(1) + 1;

        // Prepare visualization frame (for both display and Rerun)
        cv::Mat view_debug = result.resized_frame_320x640.clone();
        float steering_angle = result.steering_angle;
        cv::Mat rotatedPredSteeringWheelImg = rotateSteeringWheel(predSteeringWheelImg, steering_angle);

        // Read GT steering from CAN frame
        std::optional<float> gtSteeringAngle;
        cv::Mat rotatedGtSteeringWheelImg;
        if (can_interface) {
            if (can_interface->getState().is_valid && can_interface->getState().is_steering_angle) {
                gtSteeringAngle = can_interface->getState().steering_angle_deg;
                if (gtSteeringAngle.has_value()) {
                    rotatedGtSteeringWheelImg = rotateSteeringWheel(gtSteeringWheelImg, gtSteeringAngle.value());
                }
            }
        }

        visualizeSteering(view_debug, steering_angle, rotatedPredSteeringWheelImg, gtSteeringAngle, rotatedGtSteeringWheelImg);
        drawRawMasksInPlace(view_debug, result.lanes);

#ifdef ENABLE_RERUN
        // Log to Rerun (independent of visualization - works even if enable_viz=false)
        if (rerun_logger && rerun_logger->isEnabled()) {
            long inference_time_us = duration_cast<microseconds>(
                result.inference_time - result.capture_time
            ).count();

            rerun_logger->logData(
                result.frame_number,
                result.resized_frame_320x640,
                result.lanes,
                view_debug,
                result.vehicle_state,
                result.steering_angle_raw,
                result.steering_angle,
                result.autosteer_angle,
                result.path_output,
                inference_time_us
            );
        }
#endif

        // Visualization
        if (enable_viz) {
            // drawPolyFitLanesInPlace(
            //     view_final,
            //     result.lanes
            // );
            //  drawBEVVis(
            //      view_bev,
            //      result.frame,
            //      result.metrics.bev_visuals
            //  );
            //
            //  // Draw Metric Debug (projected back to pixels) - only if path is valid
            //  if (result.path_output.fused_valid) {
            //      std::vector<double> left_coeffs(result.path_output.left_coeff.begin(), result.path_output.left_coeff.end());
            //      std::vector<double> right_coeffs(result.path_output.right_coeff.begin(), result.path_output.right_coeff.end());
            //      autoware_pov::vision::egolanes::drawMetricVerification(
            //          view_bev,
            //          left_coeffs,
            //          right_coeffs
            //      );
            //  }
            //
            //  // 3. View layout handling
            //  // Layout:
            //  // | [Debug] | [ BEV (640x640) ]
            //  // | [Final] | [ Black Space   ]
            //
            //  // Left col: debug (top) + final (bottom)
            //  cv::Mat left_col;
            // cv::vconcat(
            //     view_debug,
            //     view_final,
            //      left_col
            //  );
            //
            //  float left_aspect = static_cast<float>(left_col.cols) / left_col.rows;
            //  int target_left_w = static_cast<int>(window_height * left_aspect);
            //  cv::resize(
            //      left_col,
            //      left_col,
            //      cv::Size(target_left_w, window_height)
            //  );
            //
            //  // Right col: BEV (stretched to match height)
            //  // Black canvas matching left col height
            //  cv::Mat right_col = cv::Mat::zeros(
            //      window_height,
            //      640,
            //      view_bev.type()
            //  );
            //  // Prep BEV
            //  cv::Rect top_roi(
            //      0, 0,
            //      view_bev.cols,
            //      view_bev.rows
            //  );
            //  view_bev.copyTo(right_col(top_roi));
            //
            //  // Final stacked view
            //  cv::Mat stacked_view;
            //  cv::hconcat(
            //      left_col,
            //      right_col,
            //     stacked_view
            // );
            //
            // // Initialize video writer on first frame
            // if (save_video && !video_writer_initialized) {
            //     // Use H.264 for better performance and smaller file size
            //     // XVID is slower and creates larger files
            //     int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
            //     video_writer.open(
            //         output_video_path,
            //         fourcc,
            //          10.0,
            //         stacked_view.size(),
            //         true
            //     );
            //
            //     if (video_writer.isOpened()) {
            //         std::cout << "Video writer initialized (H.264): " << stacked_view.cols
            //                    << "x" << stacked_view.rows << " @ 10 fps" << std::endl;
            //         video_writer_initialized = true;
            //     } else {
            //         std::cerr << "Warning: Failed to initialize video writer" << std::endl;
            //     }
            // }
            //
            // // Write to video
            // if (save_video && video_writer_initialized && video_writer.isOpened()) {
            //     video_writer.write(stacked_view);
            // }

            // Display
            cv::imshow("EgoLanes Inference", view_debug);

            if (cv::waitKey(1) == 'q') {
                running.store(false);
                break;
            }
        }

         // CSV logging: Log all frames to ensure PID and AutoSteer are captured
         // Use 0.0 for invalid PathFinder values (can be filtered in post-processing)
         if (csv_file.is_open()) {
             // Timestamp calc, from captured time
             auto ms_since_epoch = duration_cast<milliseconds>(
                 result.capture_time.time_since_epoch()
             ).count();

             csv_file << result.frame_number << ","
                      << ms_since_epoch << ","
                      // Orig metrics (for reference, but not used for tuning)
                      << result.metrics.orig_lane_offset << ","
                      << result.metrics.orig_yaw_offset << ","
                      << result.metrics.orig_curvature << ","
                      // PathFinder filtered metrics (NaN or 0.0 when invalid)
                      << (result.path_output.fused_valid ? result.path_output.cte : 0.0) << ","
                      << (result.path_output.fused_valid ? result.path_output.yaw_error : 0.0) << ","
                      << (result.path_output.fused_valid ? result.path_output.curvature : 0.0) << ","
                      // PID Controller steering angles (all in degrees)
                      << std::fixed << std::setprecision(6) << result.steering_angle_raw << ","
                      << result.steering_angle << ","
                      // AutoSteer steering angle (degrees) and validity
                      << result.autosteer_angle << ","
                      << (result.autosteer_valid ? 1 : 0) << "\n";
        }

        auto t_display_end = steady_clock::now();

        // Calculate latencies
        long display_us = duration_cast<microseconds>(
            t_display_end - t_display_start).count();
        long end_to_end_us = duration_cast<microseconds>(
            t_display_end - result.capture_time).count();

        metrics.total_display_us.fetch_add(display_us);
        metrics.total_end_to_end_us.fetch_add(end_to_end_us);

        // Print metrics every 30 frames
        if (metrics.measure_latency && count % 30 == 0) {
            long avg_capture = metrics.total_capture_us.load() / count;
            long avg_inference = metrics.total_inference_us.load() / count;
            long avg_display = metrics.total_display_us.load() / count;
            long avg_e2e = metrics.total_end_to_end_us.load() / count;

            std::cout << "\n========================================\n";
            std::cout << "Frames processed: " << count << "\n";
            std::cout << "Pipeline Latencies:\n";
            std::cout << "  1. Capture:       " << std::fixed << std::setprecision(2)
                     << (avg_capture / 1000.0) << " ms\n";
            std::cout << "  2. Inference:     " << (avg_inference / 1000.0)
                     << " ms (" << (1000000.0 / avg_inference) << " FPS capable)\n";
            std::cout << "  3. Display:       " << (avg_display / 1000.0) << " ms\n";
            std::cout << "  4. End-to-End:    " << (avg_e2e / 1000.0) << " ms\n";
            std::cout << "Throughput: " << (count / (avg_e2e * count / 1000000.0)) << " FPS\n";
            std::cout << "========================================\n";
        }
    }

     // Cleanups

     // Video writer
    if (save_video && video_writer_initialized && video_writer.isOpened()) {
        video_writer.release();
        std::cout << "\nVideo saved to: " << output_video_path << std::endl;
    }

     // Vis
    if (enable_viz) {
        cv::destroyAllWindows();
    }

     // CSV logger
     if (csv_file.is_open()) {
         csv_file.close();
         std::cout << "CSV log saved." << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::string config_path = (argc >= 2) ? argv[1] : "/usr/share/visionpilot/visionpilot.conf";
    config = ConfigReader::loadFromFile(config_path);
    std::cout << "Loaded configuration from: " << config_path << std::endl;

    // Extract configuration values
    std::string mode = config.mode;
    std::string source;
    bool is_camera = (mode == "camera");

    if (is_camera) {
        // Interactive camera selection
        if (config.source.camera_auto_select) {
            source = selectCamera();
            if (source.empty()) {
                std::cout << "No camera selected. Exiting." << std::endl;
                return 0;
            }
        } else {
            source = config.source.camera_device_id;
        }

        // Verify camera works
        if (!config.simulation && !verifyCamera(source)) {
            std::cerr << "\nCamera verification failed." << std::endl;
            std::cerr << "Please check connection and driver installation." << std::endl;
            printDriverInstructions();
            return 1;
        }
    } else {
        source = config.source.video_path;
    }

    std::string model_path = config.models.egolanes_path;
    std::string provider = config.models.provider;
    std::string precision = config.models.precision;
    int device_id = config.models.device_id;
    std::string cache_dir = config.models.cache_dir;
    float threshold = config.models.threshold;

    bool measure_latency = config.output.measure_latency;
    bool enable_viz = config.output.enable_viz;
    bool save_video = config.output.save_video;
    std::string output_video_path = config.output.output_video_path;
    std::string csv_log_path = config.output.csv_log_path;

    bool enable_rerun = config.rerun.enabled;
    bool spawn_rerun_viewer = config.rerun.spawn_viewer;
    std::string rerun_save_path = config.rerun.save_path;

    std::string autosteer_model_path = config.models.autosteer_path;

    std::string can_interface_name = "";
    if (config.can_interface.enabled) {
        can_interface_name = config.can_interface.interface_name;
    }

    double K_p = config.steering_control.Kp;
    double K_i = config.steering_control.Ki;
    double K_d = config.steering_control.Kd;
    double K_S = config.steering_control.Ks;

    if (save_video && !enable_viz) {
        std::cerr << "Warning: save_video requires enable_viz=true. Enabling visualization." << std::endl;
        enable_viz = true;
    }

    // Initialize inference backend
    std::cout << "Loading model: " << model_path << std::endl;
#ifdef SKIP_ORT
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
    std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
#else
#ifdef SKIP_ORT
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
    std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
#else
    std::cout << "Provider: " << provider << " | Precision: " << precision << std::endl;

    if (provider == "tensorrt") {
        std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
        std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
    }
#endif
#endif

#ifdef SKIP_ORT
    // TensorRT: precision is fp16 or fp32, no provider needed
    EgoLanesEngine engine(model_path, precision, device_id, cache_dir);
#else
    // ONNX Runtime: provider and precision
    EgoLanesEngine engine(model_path, provider, precision, device_id, cache_dir);
#endif
    std::cout << "Backend initialized!\n" << std::endl;

    // Warm-up inference (builds TensorRT engine on first run)
#ifdef SKIP_ORT
    // TensorRT always needs warm-up
    std::cout << "Running warm-up inference to build TensorRT engine..." << std::endl;
    std::cout << "This may take 20-60 seconds on first run. Please wait...\n" << std::endl;

    cv::Mat dummy_frame(720, 1280, CV_8UC3, cv::Scalar(128, 128, 128));
    auto warmup_start = std::chrono::steady_clock::now();

    // Run warm-up inference
    LaneSegmentation warmup_result = engine.inference(dummy_frame, threshold);

    auto warmup_end = std::chrono::steady_clock::now();
    double warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        warmup_end - warmup_start).count() / 1000.0;

    std::cout << "Warm-up complete! (took " << std::fixed << std::setprecision(1)
              << warmup_time << "s)" << std::endl;
    std::cout << "TensorRT engine is now cached and ready.\n" << std::endl;
#else
    // ONNX Runtime: only warm-up if using TensorRT provider
    if (provider == "tensorrt") {
        std::cout << "Running warm-up inference to build TensorRT engine..." << std::endl;
        std::cout << "This may take 20-60 seconds on first run. Please wait...\n" << std::endl;

        cv::Mat dummy_frame(720, 1280, CV_8UC3, cv::Scalar(128, 128, 128));
        auto warmup_start = std::chrono::steady_clock::now();

        // Run warm-up inference
        LaneSegmentation warmup_result = engine.inference(dummy_frame, threshold);

        auto warmup_end = std::chrono::steady_clock::now();
        double warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            warmup_end - warmup_start).count() / 1000.0;

        std::cout << "Warm-up complete! (took " << std::fixed << std::setprecision(1)
                  << warmup_time << "s)" << std::endl;
        std::cout << "TensorRT engine is now cached and ready.\n" << std::endl;
    }
#endif

    std::cout << "Backend ready!\n" << std::endl;

#ifdef ENABLE_RERUN
    // Initialize Rerun logger (optional)
    std::unique_ptr<autoware_pov::vision::rerun_integration::RerunLogger> rerun_logger;
    if (enable_rerun) {
        rerun_logger = std::make_unique<autoware_pov::vision::rerun_integration::RerunLogger>(
            "EgoLanes", spawn_rerun_viewer, rerun_save_path);
    }
#endif

    // Initialize PathFinder (mandatory - uses LaneTracker's BEV output)
    std::unique_ptr<PathFinder> path_finder = std::make_unique<PathFinder>(4.0);  // 4.0m default lane width
    std::cout << "PathFinder initialized (Bayes filter + polynomial fitting)" << std::endl;
    std::cout << "  - Using BEV points from LaneTracker" << std::endl;
    std::cout << "  - Transform: BEV pixels → meters (TODO: calibrate)" << "\n" << std::endl;

    // Initialize Steering Controller (mandatory)
    std::unique_ptr<SteeringController> steering_controller = std::make_unique<SteeringController>(K_p, K_i, K_d, K_S);
    std::cout << "Steering Controller initialized" << std::endl;

    // Initialize AutoSteer (mandatory - temporal steering prediction)
    std::unique_ptr<AutoSteerEngine> autosteer_engine;
    std::cout << "\nLoading AutoSteer model: " << autosteer_model_path << std::endl;

#ifdef SKIP_ORT
    // TensorRT Direct: precision is fp16 or fp32, no provider needed
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
    std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;

    autosteer_engine = std::make_unique<AutoSteerEngine>(
        autosteer_model_path, precision, device_id, cache_dir);

    // Warm-up AutoSteer inference (builds TensorRT engine on first run)
    std::cout << "Running AutoSteer warm-up inference to build TensorRT engine..." << std::endl;
    std::cout << "This may take 20-60 seconds on first run. Please wait...\n" << std::endl;

    auto autosteer_warmup_start = std::chrono::steady_clock::now();
    std::vector<float> dummy_input(6 * 80 * 160, 0.5f);
    float autosteer_warmup_result = autosteer_engine->inference(dummy_input);
    auto autosteer_warmup_end = std::chrono::steady_clock::now();
    double autosteer_warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        autosteer_warmup_end - autosteer_warmup_start).count() / 1000.0;

    std::cout << "AutoSteer warm-up complete! (took " << std::fixed << std::setprecision(1)
              << autosteer_warmup_time << "s)" << std::endl;
    std::cout << "TensorRT engine is now cached and ready.\n" << std::endl;
#else
    // ONNX Runtime: provider and precision
    std::cout << "Provider: " << provider << " | Precision: " << precision << std::endl;

    if (provider == "tensorrt") {
        std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
        std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
    }

    autosteer_engine = std::make_unique<AutoSteerEngine>(
        autosteer_model_path, provider, precision, device_id, cache_dir);

    // Warm-up AutoSteer inference (builds TensorRT engine on first run)
    if (provider == "tensorrt") {
        std::cout << "Running AutoSteer warm-up inference to build TensorRT engine..." << std::endl;
        std::cout << "This may take 20-60 seconds on first run. Please wait...\n" << std::endl;

        auto autosteer_warmup_start = std::chrono::steady_clock::now();
        std::vector<float> dummy_input(6 * 80 * 160, 0.5f);
        float autosteer_warmup_result = autosteer_engine->inference(dummy_input);
        auto autosteer_warmup_end = std::chrono::steady_clock::now();
        double autosteer_warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            autosteer_warmup_end - autosteer_warmup_start).count() / 1000.0;

        std::cout << "AutoSteer warm-up complete! (took " << std::fixed << std::setprecision(1)
                  << autosteer_warmup_time << "s)" << std::endl;
        std::cout << "TensorRT engine is now cached and ready.\n" << std::endl;
    }
#endif

    std::cout << "AutoSteer initialized (temporal steering prediction)" << std::endl;
    std::cout << "  - Input: [1, 6, 80, 160] (concatenated EgoLanes t-1, t)" << std::endl;
    std::cout << "  - Output: Steering angle (degrees, -30 to +30)" << std::endl;
    std::cout << "  - Note: First frame will be skipped (requires temporal buffer)\n" << std::endl;

    // Initialize CAN Interface (optional - ground truth)
    std::unique_ptr<CanInterface> can_interface;
    if (!can_interface_name.empty()) {
        try {
            can_interface = std::make_unique<CanInterface>(can_interface_name);
            std::cout << "CAN Interface initialized: " << can_interface_name << std::endl;
        } catch (...) {
            std::cerr << "Warning: Failed to initialize CAN interface '" << can_interface_name
                      << "'. Continuing without CAN data." << std::endl;
        }
    }

    // Thread-safe queues with bounded size (prevents memory overflow)
    ThreadSafeQueue<TimestampedFrame> capture_queue(5);   // Max 5 frames waiting for inference
    ThreadSafeQueue<InferenceResult> display_queue(5);    // Max 5 frames waiting for display

    // Performance metrics
    PerformanceMetrics metrics;
    metrics.measure_latency = measure_latency;
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "========================================" << std::endl;
    std::cout << "Starting multi-threaded inference pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Source: " << (is_camera ? "Camera" : "Video") << std::endl;
    std::cout << "Mode: " << (enable_viz ? "Visualization" : "Headless") << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
#ifdef ENABLE_RERUN
    if (enable_rerun && rerun_logger && rerun_logger->isEnabled()) {
        std::cout << "Rerun logging: ENABLED" << std::endl;
    }
#endif
    std::cout << "PathFinder: ENABLED (polynomial fitting + Bayes filter)" << std::endl;
    std::cout << "Steering Control: ENABLED" << std::endl;
    std::cout << "AutoSteer: ENABLED (temporal steering prediction)" << std::endl;
    if (can_interface) {
        std::cout << "CAN Interface: ENABLED (Ground Truth)" << std::endl;
    }
    if (measure_latency) {
        std::cout << "Latency measurement: ENABLED (metrics every 30 frames)" << std::endl;
    }
    if (save_video && enable_viz) {
        std::cout << "Video saving: ENABLED -> " << output_video_path << std::endl;
    }
    if (enable_viz) {
        std::cout << "Press 'q' in the video window to quit" << std::endl;
    } else {
        std::cout << "Running in headless mode" << std::endl;
        std::cout << "Press Ctrl+C to quit" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    std::thread t_capture(captureThread, source, is_camera, std::ref(capture_queue),
                          std::ref(metrics), std::ref(running), can_interface.get());
    std::thread t_inference(inferenceThread, std::ref(engine),
                            std::ref(capture_queue), std::ref(display_queue),
                            std::ref(metrics), std::ref(running), threshold,
                            path_finder.get(),
                            steering_controller.get(),
                            autosteer_engine.get());
#ifdef ENABLE_RERUN
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics),
                          std::ref(running), enable_viz, save_video, output_video_path, csv_log_path,
                          rerun_logger.get());
#else
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics),
                          std::ref(running), enable_viz, save_video, output_video_path, csv_log_path);
#endif

    // Wait for threads
    t_capture.join();
    t_inference.join();
    t_display.join();

    std::cout << "\nInference pipeline stopped." << std::endl;

    return 0;
}
