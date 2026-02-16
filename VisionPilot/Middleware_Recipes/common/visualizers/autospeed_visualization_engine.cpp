#include "../include/autospeed_visualization_engine.hpp"
#include <sstream>
#include <iomanip>

namespace autoware_pov::common {

// Static color definitions (by level: 1=red, 2=yellow, 3=cyan)
const cv::Scalar AutoSpeedVisualizationEngine::LEVEL_COLORS[] = {
    cv::Scalar(255, 255, 255),  // Level 0: White (fallback)
    cv::Scalar(0, 0, 255),      // Level 1: Red (BGR) - Main CIPO priority
    cv::Scalar(0, 255, 255),    // Level 2: Yellow (BGR) - Secondary priority
    cv::Scalar(255, 255, 0),    // Level 3: Cyan (BGR) - Other
};
const cv::Scalar AutoSpeedVisualizationEngine::CIPO_COLOR = cv::Scalar(0, 255, 0);  // Green
const cv::Scalar AutoSpeedVisualizationEngine::WARNING_COLOR = cv::Scalar(0, 0, 255);  // Red
const cv::Scalar AutoSpeedVisualizationEngine::KALMAN_RESET_COLOR = cv::Scalar(0, 165, 255);  // Orange

AutoSpeedVisualizationEngine::AutoSpeedVisualizationEngine(bool show_opencv_window)
    : show_opencv_window_(show_opencv_window)
    , opencv_frame_count_(0)
    , opencv_fps_(0.0)
    , cut_in_frames_remaining_(0)
    , kalman_reset_frames_remaining_(0) {
    last_opencv_time_ = std::chrono::steady_clock::now();
}

cv::Scalar AutoSpeedVisualizationEngine::getColorByClass(int class_id) {
    if (class_id >= 0 && class_id <= 3) {
        return LEVEL_COLORS[class_id];
    }
    return LEVEL_COLORS[0];  // White fallback
}

bool AutoSpeedVisualizationEngine::isDetectionTracked(
    const cv::Rect& bbox,
    const std::vector<vision::TrackedObject>& tracked_objects) {

    for (const auto& obj : tracked_objects) {
        // Check if bboxes overlap significantly (IoU > 0.5)
        int x_overlap = std::max(0, std::min(bbox.x + bbox.width, obj.bbox.x + obj.bbox.width) -
                                    std::max(bbox.x, obj.bbox.x));
        int y_overlap = std::max(0, std::min(bbox.y + bbox.height, obj.bbox.y + obj.bbox.height) -
                                    std::max(bbox.y, obj.bbox.y));
        int overlap_area = x_overlap * y_overlap;
        int bbox_area = bbox.width * bbox.height;
        int obj_area = obj.bbox.width * obj.bbox.height;
        int union_area = bbox_area + obj_area - overlap_area;

        if (union_area > 0 && static_cast<float>(overlap_area) / union_area > 0.5f) {
            return true;
        }
    }
    return false;
}

cv::Mat AutoSpeedVisualizationEngine::visualize(
    const cv::Mat& frame,
    const std::vector<vision::autospeed::Detection>& detections,
    const std::vector<vision::TrackedObject>& tracked_objects,
    const vision::CIPOInfo& cipo,
    bool cut_in_detected,
    bool kalman_reset) {

    cv::Mat display_frame = frame.clone();

    // STEP 1: Draw untracked detections (lighter overlay, thin border)
    for (const auto& det : detections) {
        cv::Rect bbox(static_cast<int>(det.x1), static_cast<int>(det.y1),
                      static_cast<int>(det.x2 - det.x1), static_cast<int>(det.y2 - det.y1));

        // Skip if this detection is tracked (will be drawn in next step)
        if (isDetectionTracked(bbox, tracked_objects)) {
            continue;
        }

        cv::Scalar color = getColorByClass(det.class_id);
        drawUntrackedDetection(display_frame, det, color);
    }

    // STEP 2: Draw tracked objects with full info
    for (const auto& obj : tracked_objects) {
        bool is_main_cipo = (cipo.exists && cipo.track_id == obj.track_id);
        cv::Scalar color = getColorByClass(obj.class_id);
        drawTrackedObject(display_frame, obj, color, is_main_cipo);
    }

    // STEP 4: Draw event warnings (cut-in, Kalman reset) with persistence
    // Reset counter when new event detected
    if (cut_in_detected) {
        cut_in_frames_remaining_ = WARNING_DISPLAY_FRAMES;
    }
    if (kalman_reset) {
        kalman_reset_frames_remaining_ = WARNING_DISPLAY_FRAMES;
    }

    // Draw warnings if counter > 0
    if (cut_in_frames_remaining_ > 0 || kalman_reset_frames_remaining_ > 0) {
        int warning_y = 80;
        if (cut_in_frames_remaining_ > 0) {
            drawCutInWarning(display_frame, warning_y);
            cut_in_frames_remaining_--;
        }
        if (kalman_reset_frames_remaining_ > 0) {
            drawKalmanResetIndicator(display_frame, warning_y);
            kalman_reset_frames_remaining_--;
        }
    }

    // Display OpenCV window if enabled
    if (show_opencv_window_) {
        displayOpenCVWindow(display_frame, detections, cipo);
    }

    return display_frame;
}

cv::Mat AutoSpeedVisualizationEngine::visualize(
    const cv::Mat& frame,
    const std::vector<vision::autospeed::Detection>& detections,
    const vision::CIPOInfo& cipo) {

    // Backward compatible: call full version with empty tracked objects
    return visualize(frame, detections, std::vector<vision::TrackedObject>(), cipo, false, false);
}

void AutoSpeedVisualizationEngine::drawUntrackedDetection(
    cv::Mat& frame,
    const vision::autospeed::Detection& det,
    const cv::Scalar& color) {

    cv::Rect bbox(static_cast<int>(det.x1), static_cast<int>(det.y1),
                  static_cast<int>(det.x2 - det.x1), static_cast<int>(det.y2 - det.y1));

    // Draw semi-transparent filled bounding box (20% overlay - lighter for untracked)
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, bbox, color, cv::FILLED);
    cv::addWeighted(overlay, 0.2, frame, 0.8, 0, frame);

    // Draw thin border for untracked detections
    cv::rectangle(frame, bbox, color, 1);

    // Simple confidence label for untracked detections
    std::stringstream label;
    label << std::fixed << std::setprecision(2) << det.confidence;
    std::string label_text = label.str();

    cv::putText(frame, label_text,
               cv::Point(bbox.x + 5, bbox.y + 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.5,
               cv::Scalar(255, 255, 255),  // White text
               1, cv::LINE_AA);
}

void AutoSpeedVisualizationEngine::drawTrackedObject(
    cv::Mat& frame,
    const vision::TrackedObject& obj,
    const cv::Scalar& color,
    bool is_main_cipo) {

    // Draw semi-transparent filled bounding box (35% overlay - darker for tracked)
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, obj.bbox, color, cv::FILLED);
    cv::addWeighted(overlay, 0.35, frame, 0.65, 0, frame);

    // Draw solid border on top (thicker for main CIPO)
    int border_thickness = is_main_cipo ? 4 : 2;
    cv::rectangle(frame, obj.bbox, color, border_thickness);

    // Prepare distance and speed text (ABOVE the bbox)
    std::stringstream label;
    label << std::fixed << std::setprecision(1);
    label << obj.distance_m << "m | " << obj.velocity_ms << "m/s";

    // Add track ID for non-CIPO objects
    if (!is_main_cipo) {
        label << " [" << obj.track_id << "]";
    }

    std::string label_text = label.str();
    int baseline = 0;
    float font_scale = is_main_cipo ? 0.8f : 0.6f;  // CIPO: 0.8, non-CIPO: 0.6
    int font_thickness = is_main_cipo ? 2 : 1;
    cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                           font_scale, font_thickness, &baseline);

    // Calculate label position ABOVE bbox
    int label_y = obj.bbox.y - 10;
    int cipo_label_height = 0;

    // For CIPO: add extra space for CIPO tag above distance label
    if (is_main_cipo) {
        cv::Size cipo_tag_size = cv::getTextSize("CIPO", cv::FONT_HERSHEY_SIMPLEX,
                                                  0.9f, 2, &baseline);
        cipo_label_height = cipo_tag_size.height + 15;
    }

    // Check if label fits above, otherwise put below
    if (label_y < label_size.height + cipo_label_height + 20) {
        label_y = obj.bbox.y + obj.bbox.height + label_size.height + 10;
    }

    // Draw label background (black)
    cv::Point label_origin(obj.bbox.x, label_y - label_size.height);
    int total_bg_height = label_size.height + 10 + cipo_label_height;
    cv::Rect label_bg(label_origin.x - 5, label_origin.y - 5 - cipo_label_height,
                     label_size.width + 10, total_bg_height);
    cv::rectangle(frame, label_bg, cv::Scalar(0, 0, 0), cv::FILLED);

    // For CIPO: draw "CIPO" tag above distance/velocity
    if (is_main_cipo) {
        std::string cipo_tag = "CIPO";
        cv::putText(frame, cipo_tag,
                   cv::Point(obj.bbox.x, label_y - label_size.height - 8),
                   cv::FONT_HERSHEY_SIMPLEX, 0.9f,
                   CIPO_COLOR,  // Green text
                   2, cv::LINE_AA);
    }

    // Draw label text (white for high contrast)
    cv::putText(frame, label_text,
               cv::Point(obj.bbox.x, label_y),
               cv::FONT_HERSHEY_SIMPLEX, font_scale,
               cv::Scalar(255, 255, 255),  // White text
               font_thickness, cv::LINE_AA);

    // Draw distance measurement point (bottom center)
    drawDistancePoint(frame, obj.bbox, color);
}

void AutoSpeedVisualizationEngine::drawDistancePoint(
    cv::Mat& frame,
    const cv::Rect& bbox,
    const cv::Scalar& color) {

    cv::Point2f bottom_center(
        bbox.x + bbox.width / 2.0f,
        bbox.y + bbox.height
    );
    cv::circle(frame, bottom_center, 5, color, -1);
    cv::circle(frame, bottom_center, 6, cv::Scalar(255, 255, 255), 2);  // White outline
}


void AutoSpeedVisualizationEngine::drawCutInWarning(cv::Mat& frame, int& warning_y) {
    std::string warning_text = "!!! CUT-IN DETECTED !!!";
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(warning_text, cv::FONT_HERSHEY_SIMPLEX,
                                         1.2, 3, &baseline);

    int warning_x = (frame.cols - text_size.width) / 2;

    // Draw black background with red border
    cv::Rect bg_rect(warning_x - 15, warning_y - text_size.height - 10,
                   text_size.width + 30, text_size.height + 20);
    cv::rectangle(frame, bg_rect, WARNING_COLOR, 4);  // Red border
    cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);  // Black bg

    // Draw warning text
    cv::putText(frame, warning_text,
               cv::Point(warning_x, warning_y),
               cv::FONT_HERSHEY_SIMPLEX, 1.2,
               WARNING_COLOR,  // Red text
               3, cv::LINE_AA);

    warning_y += text_size.height + 35;
}

void AutoSpeedVisualizationEngine::drawKalmanResetIndicator(cv::Mat& frame, int& warning_y) {
    std::string reset_text = "Kalman Filter Reset";
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(reset_text, cv::FONT_HERSHEY_SIMPLEX,
                                         0.9, 2, &baseline);

    int reset_x = (frame.cols - text_size.width) / 2;

    // Draw black background with orange border
    cv::Rect bg_rect(reset_x - 10, warning_y - text_size.height - 8,
                   text_size.width + 20, text_size.height + 16);
    cv::rectangle(frame, bg_rect, KALMAN_RESET_COLOR, 3);  // Orange border
    cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);  // Black bg

    // Draw reset text
    cv::putText(frame, reset_text,
               cv::Point(reset_x, warning_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.9,
               KALMAN_RESET_COLOR,  // Orange text
               2, cv::LINE_AA);

    warning_y += text_size.height + 30;
}

void AutoSpeedVisualizationEngine::displayOpenCVWindow(
    const cv::Mat& image,
    const std::vector<vision::autospeed::Detection>& detections,
    const vision::CIPOInfo& cipo) {

    // Measure OpenCV display FPS
    auto current_time = std::chrono::steady_clock::now();
    opencv_frame_count_++;

    // Calculate FPS every 30 frames for smooth display
    if (opencv_frame_count_ % 30 == 0) {
        auto time_diff = std::chrono::duration<double>(current_time - last_opencv_time_).count();
        opencv_fps_ = 30.0 / time_diff;
        last_opencv_time_ = current_time;
    }

    // Add performance metrics overlay
    cv::Mat display_image = image.clone();

    // Performance metrics text
    std::string opencv_fps_text = "FPS: " + std::to_string(static_cast<int>(opencv_fps_));
    std::string detections_text = "Detections: " + std::to_string(detections.size());
    std::string cipo_text;
    if (cipo.exists) {
        std::stringstream ss;
        ss << "CIPO: ID=" << cipo.track_id
           << " | " << std::fixed << std::setprecision(1) << cipo.distance_m << "m"
           << " | " << std::fixed << std::setprecision(1) << cipo.velocity_ms << "m/s";
        cipo_text = ss.str();
    } else {
        cipo_text = "No CIPO";
    }

    // Position text in top-left corner
    int line_height = 40;

    cv::putText(display_image, cipo_text, cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::putText(display_image, opencv_fps_text, cv::Point(10, 50 + line_height),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(display_image, detections_text, cv::Point(10, 50 + 2 * line_height),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

    // Display window
    cv::namedWindow("AutoSpeed Detections", cv::WINDOW_NORMAL);
    cv::resizeWindow("AutoSpeed Detections", 1280, 720);
    cv::imshow("AutoSpeed Detections", display_image);
    cv::waitKey(1);  // Non-blocking display update
}

} // namespace autoware_pov::common
