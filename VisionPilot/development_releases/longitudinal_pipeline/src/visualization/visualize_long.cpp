// Draw detections and tracked objects with IDs, distances, velocities, and CIPO indicator

#include "../../include/visualization/visualize_long.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "../../include/tracking/object_finder.hpp"
#include "../../include/inference/autospeed/onnxruntime_engine.hpp"

using namespace autoware_pov::vision::tracking;
using namespace autoware_pov::vision::autospeed;

void drawTrackedObjects(cv::Mat& frame,
                        const std::vector<Detection>& detections,
                        const std::vector<TrackedObject>& tracked_objects,
                        const CIPOInfo& cipo,
                        bool cut_in_detected,
                        bool kalman_reset)
{
    // Color map based on level (1=red, 2=yellow, 3=cyan)
    auto getColor = [](int class_id) -> cv::Scalar {
        switch(class_id) {
            case 1: return cv::Scalar(0, 0, 255);      // Red (BGR) - Level 1 (Main CIPO priority)
            case 2: return cv::Scalar(0, 255, 255);    // Yellow (BGR) - Level 2 (Secondary priority)
            case 3: return cv::Scalar(255, 255, 0);    // Cyan (BGR) - Level 3 (Other)
            default: return cv::Scalar(255, 255, 255); // White fallback
        }
    };

    // Helper to check if a detection bbox is being tracked
    auto isTracked = [&tracked_objects](const cv::Rect& bbox) -> bool {
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
            
            if (union_area > 0 && (float)overlap_area / union_area > 0.5f) {
                return true;
            }
        }
        return false;
    };

    // STEP 1: Draw ALL detections (especially untracked level 3s) as basic boxes
    for (const auto& det : detections) {
        cv::Rect bbox(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        
        // Skip if this detection is already tracked (will be drawn in next step)
        if (isTracked(bbox)) {
            continue;
        }
        
        cv::Scalar color = getColor(det.class_id);
        
        // Draw semi-transparent filled bounding box (lighter for untracked)
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, bbox, color, cv::FILLED);
        cv::addWeighted(overlay, 0.2, frame, 0.8, 0, frame);  // 20% overlay (lighter)
        
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

    // STEP 2: Draw TRACKED objects with full info (distance, velocity, track ID)
    for (const auto& obj : tracked_objects) {
        bool is_main_cipo = (cipo.exists && cipo.track_id == obj.track_id);
        
        cv::Scalar color = getColor(obj.class_id);
        
        // Draw semi-transparent filled bounding box (darker for tracked)
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, obj.bbox, color, cv::FILLED);
        cv::addWeighted(overlay, 0.35, frame, 0.65, 0, frame);  // 35% overlay (darker for tracked)
        
        // Draw solid border on top (thicker for main_CIPO)
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
        float font_scale = is_main_cipo ? 0.8 : 0.6;
        int font_thickness = is_main_cipo ? 2 : 1;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 
                                               font_scale, font_thickness, &baseline);
        
        // Position label ABOVE bbox
        int label_y = obj.bbox.y - 10;
        if (label_y < label_size.height + 10) {
            label_y = obj.bbox.y + obj.bbox.height + label_size.height + 10;
        }
        
        // Draw label background (black with slight transparency)
        cv::Point label_origin(obj.bbox.x, label_y - label_size.height);
        cv::Rect label_bg(label_origin.x - 5, label_origin.y - 5, 
                         label_size.width + 10, label_size.height + 10);
        cv::rectangle(frame, label_bg, cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Draw label text (white for high contrast)
        cv::putText(frame, label_text,
                   cv::Point(obj.bbox.x, label_y),
                   cv::FONT_HERSHEY_SIMPLEX, font_scale,
                   cv::Scalar(255, 255, 255),  // White text
                   font_thickness, cv::LINE_AA);
        
        // Draw bottom-center point (where distance is measured)
        cv::Point2f bottom_center(
            obj.bbox.x + obj.bbox.width / 2.0f,
            obj.bbox.y + obj.bbox.height
        );
        cv::circle(frame, bottom_center, 5, color, -1);
        cv::circle(frame, bottom_center, 6, cv::Scalar(255, 255, 255), 2);  // White outline
    }
    
    // Draw main_CIPO summary in top-left corner (ALWAYS shown when exists)
    if (cipo.exists) {
        std::stringstream cipo_text;
        cipo_text << "main_CIPO: Track " << cipo.track_id 
                  << " (Level " << cipo.class_id << ") "
                  << std::fixed << std::setprecision(1)
                  << cipo.distance_m << "m, "
                  << cipo.velocity_ms << "m/s";
        
        std::string text = cipo_text.str();
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                             0.7, 2, &baseline);
        
        // Draw background
        cv::Rect bg_rect(5, 5, text_size.width + 10, text_size.height + 10);
        cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Draw text
        cv::putText(frame, text,
                   cv::Point(10, 15 + text_size.height),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 0),  // Green text for CIPO
                   2, cv::LINE_AA);
    } else {
        // No main_CIPO message
        std::string text = "No main_CIPO detected";
        cv::putText(frame, text,
                   cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 0, 255),  // Red text
                   2, cv::LINE_AA);
    }
    
    // ===== EVENT WARNINGS =====
    // Display prominent warnings for cut-in detection and Kalman reset
    if (cut_in_detected || kalman_reset) {
        // Position warnings in center-top of frame
        int warning_y = 80;
        
        if (cut_in_detected) {
            std::string warning_text = "!!! CUT-IN DETECTED !!!";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(warning_text, cv::FONT_HERSHEY_SIMPLEX, 
                                                 1.2, 3, &baseline);
            
            int warning_x = (frame.cols - text_size.width) / 2;
            
            // Draw black background with red border
            cv::Rect bg_rect(warning_x - 15, warning_y - text_size.height - 10, 
                           text_size.width + 30, text_size.height + 20);
            cv::rectangle(frame, bg_rect, cv::Scalar(255, 0, 0), 4);  // Red border
            cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);  // Black bg
            
            // Draw warning text
            cv::putText(frame, warning_text,
                       cv::Point(warning_x, warning_y),
                       cv::FONT_HERSHEY_SIMPLEX, 1.2,
                       cv::Scalar(0, 0, 255),  // Red text
                       3, cv::LINE_AA);
            
            warning_y += text_size.height + 35;
        }
        
        if (kalman_reset) {
            std::string reset_text = "Kalman Filter Reset";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(reset_text, cv::FONT_HERSHEY_SIMPLEX, 
                                                 0.9, 2, &baseline);
            
            int reset_x = (frame.cols - text_size.width) / 2;
            
            // Draw black background with orange border
            cv::Rect bg_rect(reset_x - 10, warning_y - text_size.height - 8, 
                           text_size.width + 20, text_size.height + 16);
            cv::rectangle(frame, bg_rect, cv::Scalar(0, 165, 255), 3);  // Orange border
            cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);  // Black bg
            
            // Draw reset text
            cv::putText(frame, reset_text,
                       cv::Point(reset_x, warning_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.9,
                       cv::Scalar(0, 165, 255),  // Orange text
                       2, cv::LINE_AA);
        }
    }
}
