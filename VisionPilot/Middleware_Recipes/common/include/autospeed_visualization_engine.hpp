#ifndef AUTOSPEED_VISUALIZATION_ENGINE_HPP_
#define AUTOSPEED_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "autospeed/detection.hpp"
#include "object_finder.hpp"

namespace autoware_pov::common {

class AutoSpeedVisualizationEngine {
public:
    explicit AutoSpeedVisualizationEngine(bool show_opencv_window = true);

    /**
     * Visualize detections, tracked objects, and CIPO on frame
     * @param frame Input image (BGR8)
     * @param detections Vector of detections from AutoSpeed model
     * @param tracked_objects Vector of tracked objects with distance/velocity
     * @param cipo CIPO information (optional, can be empty)
     * @param cut_in_detected Show cut-in warning banner
     * @param kalman_reset Show Kalman reset indicator
     * @return Frame with visualized detections and CIPO
     */
    cv::Mat visualize(const cv::Mat& frame,
                      const std::vector<vision::autospeed::Detection>& detections,
                      const std::vector<vision::TrackedObject>& tracked_objects,
                      const vision::CIPOInfo& cipo = vision::CIPOInfo(),
                      bool cut_in_detected = false,
                      bool kalman_reset = false);

    /**
     * Simplified visualize (backward compatible, detections only)
     */
    cv::Mat visualize(const cv::Mat& frame,
                      const std::vector<vision::autospeed::Detection>& detections,
                      const vision::CIPOInfo& cipo = vision::CIPOInfo());

private:
    /**
     * Draw untracked detection with lighter overlay
     */
    void drawUntrackedDetection(cv::Mat& frame, const vision::autospeed::Detection& det,
                                const cv::Scalar& color);

    /**
     * Draw tracked object with full info (distance, velocity, track ID)
     */
    void drawTrackedObject(cv::Mat& frame, const vision::TrackedObject& obj,
                           const cv::Scalar& color, bool is_main_cipo);

    /**
     * Draw distance measurement point (bottom center)
     */
    void drawDistancePoint(cv::Mat& frame, const cv::Rect& bbox, const cv::Scalar& color);

    /**
     * Draw cut-in warning banner
     */
    void drawCutInWarning(cv::Mat& frame, int& warning_y);

    /**
     * Draw Kalman reset indicator
     */
    void drawKalmanResetIndicator(cv::Mat& frame, int& warning_y);

    /**
     * Display OpenCV window with performance metrics overlay
     */
    void displayOpenCVWindow(const cv::Mat& image,
                             const std::vector<vision::autospeed::Detection>& detections,
                             const vision::CIPOInfo& cipo);

    /**
     * Check if a detection bbox is being tracked
     */
    bool isDetectionTracked(const cv::Rect& bbox,
                            const std::vector<vision::TrackedObject>& tracked_objects);

    /**
     * Get color based on class ID (level)
     */
    cv::Scalar getColorByClass(int class_id);

    // Color palette for different classes (by level)
    static const cv::Scalar LEVEL_COLORS[];
    static const cv::Scalar CIPO_COLOR;
    static const cv::Scalar WARNING_COLOR;
    static const cv::Scalar KALMAN_RESET_COLOR;

    // Configuration
    bool show_opencv_window_;

    // OpenCV FPS measurement
    std::chrono::steady_clock::time_point last_opencv_time_;
    int opencv_frame_count_;
    double opencv_fps_;

    // Warning display persistence (frames remaining)
    static constexpr int WARNING_DISPLAY_FRAMES = 15;
    int cut_in_frames_remaining_;
    int kalman_reset_frames_remaining_;
};

} // namespace autoware_pov::common

#endif // AUTOSPEED_VISUALIZATION_ENGINE_HPP_
