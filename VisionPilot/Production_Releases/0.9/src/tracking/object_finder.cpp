#include "tracking/object_finder.hpp"
#include "tracking/cipo_utils.hpp"
#include "tracking/logging.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace autoware_pov::vision::tracking {

ObjectFinder::ObjectFinder(const std::string& homography_yaml,
                           int image_width,
                           int image_height,
                           bool debug_mode)
    : next_track_id_(0),
      image_width_(image_width),
      image_height_(image_height),
      matching_threshold_(0.55f),
      max_frames_unmatched_(3),
      feature_match_threshold_(0.3f),
      debug_mode_(debug_mode),
      cut_in_detected_(false),
      kalman_reset_(false),
      cipo_history_(30) {  // Store last 30 frames of CIPO history
    
    // Load homography matrix from YAML
    YAML::Node config = YAML::LoadFile(homography_yaml);
    
    if (!config["H"]) {
        throw std::runtime_error("No 'H' field found in YAML file");
    }

    const auto& h_node = config["H"];
    std::vector<double> H_data;

    if (h_node.IsSequence()) {
        // Handle flat list format: H: [a, b, c, d, e, f, g, h, i]
        H_data = h_node.as<std::vector<double>>();
    } else {
        // Handle structured format: H: { rows: 3, cols: 3, data: [...] }
        H_data = h_node["data"].as<std::vector<double>>();
    }
    
    if (H_data.size() != 9) {
        throw std::runtime_error("Homography matrix must have exactly 9 elements");
    }

    H_ = cv::Mat(3, 3, CV_64F, H_data.data()).clone();
    H_.convertTo(H_, CV_32F);
                LOG_INFO(("Loaded homography matrix from " + homography_yaml).c_str());
    LOG_INFO(("Image dimensions for tracking: " + std::to_string(image_width_) + 
              "x" + std::to_string(image_height_)).c_str());
}

cv::Point2f ObjectFinder::imageToWorld(const cv::Point2f& image_point) {
    // Use OpenCV's perspectiveTransform for homography projection
    std::vector<cv::Point2f> src = {image_point};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H_);
    return dst[0];
}

float ObjectFinder::calculateDistance(const cv::Point2f& world_point) {
    // Return Euclidean distance (matches Python implementation)
    return std::sqrt(world_point.x * world_point.x + world_point.y * world_point.y);
}

std::vector<std::pair<int, int>> ObjectFinder::associateDetections(
    const std::vector<autospeed::Detection>& detections
) {
    // Result: pairs of (detection_idx, track_idx)
    // -1 means unmatched
    std::vector<std::pair<int, int>> associations;
    
    // Track which detections and tracks have been matched
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(previous_objects_.size(), false);
    
    // Greedy matching: find best match for each detection
    for (size_t det_idx = 0; det_idx < detections.size(); det_idx++) {
        const auto& det = detections[det_idx];
        
        // Only match trackable classes (1 and 2)
        if (!shouldTrackClass(det.class_id)) {
            continue;
        }
        
        // Create detection bbox
        cv::Rect det_bbox(
            static_cast<int>(det.x1),
            static_cast<int>(det.y1),
            static_cast<int>(det.x2 - det.x1),
            static_cast<int>(det.y2 - det.y1)
        );
        
        // Find best matching track
        int best_track_idx = -1;
        float best_score = 0.0f;
        
        for (size_t track_idx = 0; track_idx < previous_objects_.size(); track_idx++) {
            const auto& track = previous_objects_[track_idx];
            
            // Skip if already matched or different class
            if (track_matched[track_idx] || track.class_id != det.class_id) {
                continue;
            }
            
            // Calculate matching score (combines IoU, centroid distance, size similarity)
            float score = TrackingUtils::calculateMatchingScore(
                det_bbox, track.bbox, image_width_, image_height_
            );
            
            if (score > matching_threshold_ && score > best_score) {
                best_score = score;
                best_track_idx = track_idx;
            }
        }
        
        // Record association
        if (best_track_idx >= 0) {
            associations.push_back({det_idx, best_track_idx});
            det_matched[det_idx] = true;
            track_matched[best_track_idx] = true;
            
            if (debug_mode_) {
                LOG_INFO(("  Association: Detection " + std::to_string(det_idx) + 
                          " <-> Track " + std::to_string(previous_objects_[best_track_idx].track_id) +
                          " (score=" + std::to_string(best_score) + ")").c_str());
            }
        } else {
            // New detection (will create new track)
            associations.push_back({det_idx, -1});
        }
    }
    
    return associations;
}

std::vector<TrackedObject> ObjectFinder::update(const std::vector<autospeed::Detection>& detections,
                                                const cv::Mat& frame) {
    auto current_time = std::chrono::steady_clock::now();
    std::vector<TrackedObject> new_tracked_objects;
    
    // DEBUG: Print all detected classes (only if debug mode enabled)
    if (debug_mode_) {
        LOG_INFO("=== ALL DETECTIONS ===");
        for (size_t i = 0; i < detections.size(); i++) {
            if (shouldTrackClass(detections[i].class_id)) {
                LOG_INFO(("  Detection " + std::to_string(i) + ": class=" + 
                          std::to_string(detections[i].class_id) + " conf=" + 
                          std::to_string(detections[i].confidence)).c_str());
            }
        }
        LOG_INFO("=== DATA ASSOCIATION ===");
    }
    
    // ===== STEP 1: Associate detections with existing tracks =====
    auto associations = associateDetections(detections);
    
    // ===== STEP 2: Process matched and new detections =====
    if (debug_mode_) {
        LOG_INFO("=== PROCESSING DETECTIONS ===");
    }
    for (const auto& [det_idx, track_idx] : associations) {
            const auto& det = detections[det_idx];
            
        // Create bbox and calculate bottom-center point
        cv::Rect bbox(
                static_cast<int>(det.x1),
                static_cast<int>(det.y1),
                static_cast<int>(det.x2 - det.x1),
                static_cast<int>(det.y2 - det.y1)
            );
            
        cv::Point2f bbox_bottom_center = TrackingUtils::getBottomCenter(bbox);
        
        // Transform to world coordinates and calculate distance
        cv::Point2f world_position = imageToWorld(bbox_bottom_center);
        float measured_distance = calculateDistance(world_position);
        
        TrackedObject obj;
        
        if (track_idx >= 0) {
            // ===== MATCHED TRACK: Update existing object =====
            obj = previous_objects_[track_idx];
            obj.frames_tracked++;
            obj.frames_unmatched = 0;  // Reset unmatched counter
            
            // Calculate dt (time difference from last update)
            float dt = std::chrono::duration<float>(current_time - obj.last_update_time).count();
            
            // KALMAN FILTER: Predict -> Update
            obj.kalman.predict(dt);
            obj.kalman.update(measured_distance);
            
            // Get filtered estimates
            obj.distance_m = obj.kalman.getPosition();
            obj.velocity_ms = obj.kalman.getVelocity();
            
            if (debug_mode_) {
                LOG_INFO(("  -> MATCHED Track " + std::to_string(obj.track_id) + 
                          " (class=" + std::to_string(obj.class_id) + ")" +
                          " | dt=" + std::to_string(dt) + "s" +
                          " | Raw: " + std::to_string(measured_distance) + "m" +
                          " | Filtered: " + std::to_string(obj.distance_m) + "m" +
                          " | Velocity: " + std::to_string(obj.velocity_ms) + "m/s").c_str());
            }
            
        } else {
            // ===== NEW TRACK: Create new object =====
            obj.track_id = next_track_id_++;
            obj.class_id = det.class_id;
            obj.confidence = det.confidence;
            obj.frames_tracked = 1;
            obj.frames_unmatched = 0;
            
            // Initialize Kalman filter with first measurement
            obj.kalman.initialize(measured_distance);
            obj.distance_m = measured_distance;
            obj.velocity_ms = 0.0f;  // Unknown velocity for new tracks
            
            if (debug_mode_) {
                LOG_INFO(("  -> NEW Track " + std::to_string(obj.track_id) + 
                          " (class=" + std::to_string(obj.class_id) + ")" +
                          " | Initial distance: " + std::to_string(measured_distance) + "m").c_str());
            }
        }
        
        // Update common fields
        obj.bbox = bbox;
        obj.confidence = det.confidence;
        obj.last_update_time = current_time;
        
        new_tracked_objects.push_back(obj);
    }
    
    // ===== STEP 3: Handle unmatched tracks (keep alive for max_frames_unmatched_) =====
    if (debug_mode_) {
        LOG_INFO("=== UNMATCHED TRACKS ===");
    }
    for (size_t track_idx = 0; track_idx < previous_objects_.size(); track_idx++) {
        auto& track = previous_objects_[track_idx];
        
        // Check if this track was matched
        bool was_matched = false;
        for (const auto& [det_idx, matched_track_idx] : associations) {
            if (matched_track_idx == static_cast<int>(track_idx)) {
                was_matched = true;
                break;
            }
        }
        
        if (!was_matched) {
            track.frames_unmatched++;
            
            if (track.frames_unmatched <= max_frames_unmatched_) {
                // Keep track alive but don't update Kalman (no measurement)
                if (debug_mode_) {
                    LOG_INFO(("  -> UNMATCHED Track " + std::to_string(track.track_id) +
                              " (class=" + std::to_string(track.class_id) + ")" +
                              " | Frames unmatched: " + std::to_string(track.frames_unmatched) +
                              " | Keeping alive").c_str());
                }
                new_tracked_objects.push_back(track);
            } else {
                // Delete track (too many unmatched frames)
                if (debug_mode_) {
                    LOG_INFO(("  -> DELETED Track " + std::to_string(track.track_id) +
                              " (class=" + std::to_string(track.class_id) + ")" +
                              " | Exceeded max unmatched frames").c_str());
                }
            }
        }
    }
    
    // Update tracked objects list
    tracked_objects_ = new_tracked_objects;
    previous_objects_ = tracked_objects_;
    
    if (debug_mode_) {
        LOG_INFO(("=== TRACKING SUMMARY: " + std::to_string(tracked_objects_.size()) + 
                  " active tracks ===").c_str());
    }
    
    return tracked_objects_;
}

CIPOInfo ObjectFinder::getCIPO(const cv::Mat& frame) {
    CIPOInfo cipo;
    cipo.exists = false;
    
    // Clear previous event flags
    cut_in_detected_ = false;
    kalman_reset_ = false;
    
    // ===== STEP 1: Find main_CIPO (THE most dangerous object) =====
    int level1_idx = CIPOUtils::findClosestByLevel(tracked_objects_, 1);
    int level2_idx = CIPOUtils::findClosestByLevel(tracked_objects_, 2);
    int main_cipo_idx = CIPOUtils::selectMainCIPO(tracked_objects_, level1_idx, level2_idx);
    
    // ===== STEP 2: No main_CIPO found =====
    if (main_cipo_idx < 0) {
        LOG_INFO("No main_CIPO detected");
        return cipo;  // Empty CIPO info
    }
    
    // ===== ALWAYS log main_CIPO selection (critical info!) =====
    const auto& selected_cipo = tracked_objects_[main_cipo_idx];
    if (debug_mode_) {
        // Verbose: Show both levels
        if (level1_idx >= 0 && level2_idx >= 0) {
            LOG_INFO(("Level 1 @ " + std::to_string(tracked_objects_[level1_idx].distance_m) + "m, " +
                      "Level 2 @ " + std::to_string(tracked_objects_[level2_idx].distance_m) + "m -> " +
                      "main_CIPO: Track " + std::to_string(selected_cipo.track_id) + 
                      " (Level " + std::to_string(selected_cipo.class_id) + ")").c_str());
        } else {
            LOG_INFO(("main_CIPO: Track " + std::to_string(selected_cipo.track_id) + 
                      " (Level " + std::to_string(selected_cipo.class_id) + 
                      ") @ " + std::to_string(selected_cipo.distance_m) + "m").c_str());
        }
    } else {
        // Clean: Just show main_CIPO
        LOG_INFO(("main_CIPO: Track " + std::to_string(selected_cipo.track_id) + 
                  " (Level " + std::to_string(selected_cipo.class_id) + 
                  ") @ " + std::to_string(selected_cipo.distance_m) + "m, " + 
                  std::to_string(selected_cipo.velocity_ms) + "m/s").c_str());
    }
    
    // ===== STEP 3: Package main_CIPO info =====
    auto& main_cipo = tracked_objects_[main_cipo_idx];
    cipo.exists = true;
    cipo.track_id = main_cipo.track_id;
    cipo.class_id = main_cipo.class_id;
    cipo.distance_m = main_cipo.distance_m;
    // NOTE: velocity copied at the END after all reset logic
    
    // ===== STEP 4: Save to history for cut-in detection =====
    CIPOSnapshot snapshot;
    snapshot.track_id = main_cipo.track_id;
    snapshot.class_id = main_cipo.class_id;
    snapshot.bbox = main_cipo.bbox;
    snapshot.distance_m = main_cipo.distance_m;
    snapshot.velocity_ms = main_cipo.velocity_ms;
    snapshot.timestamp = main_cipo.last_update_time;
    snapshot.frame_crop = FeatureMatchingUtils::extractSafeCrop(frame, main_cipo.bbox);
    cipo_history_.push(snapshot);
        
    // ===== STEP 5: Detect main_CIPO change (cut-in detection) =====
    if (!cipo_history_.didCIPOChange()) {
        // No change - copy current velocity and return
        cipo.velocity_ms = main_cipo.velocity_ms;
        return cipo;
    }
    
    const CIPOSnapshot* prev_main_cipo = cipo_history_.getPrevious();
    const CIPOSnapshot* curr_main_cipo = cipo_history_.getLatest();
    
    LOG_INFO(("main_CIPO CHANGED: Track " + std::to_string(prev_main_cipo->track_id) + 
              " (Level " + std::to_string(prev_main_cipo->class_id) + ", " + 
              std::to_string(prev_main_cipo->distance_m) + "m) -> Track " + 
              std::to_string(curr_main_cipo->track_id) + 
              " (Level " + std::to_string(curr_main_cipo->class_id) + ", " + 
              std::to_string(curr_main_cipo->distance_m) + "m)").c_str());
    
    // ===== STEP 6: Use ORB to verify if it's the same vehicle =====
    if (prev_main_cipo->frame_crop.empty() || curr_main_cipo->frame_crop.empty()) {
        if (debug_mode_) {
            LOG_INFO("  -> Feature matching skipped (no image data)");
        }
        // Copy current velocity before returning
        cipo.velocity_ms = main_cipo.velocity_ms;
        return cipo;
    }
    
    bool is_same_vehicle = FeatureMatchingUtils::areSameObject(
        prev_main_cipo->frame_crop, 
        cv::Rect(0, 0, prev_main_cipo->frame_crop.cols, prev_main_cipo->frame_crop.rows),
        curr_main_cipo->frame_crop, 
        cv::Rect(0, 0, curr_main_cipo->frame_crop.cols, curr_main_cipo->frame_crop.rows),
        feature_match_threshold_
    );
    
    if (is_same_vehicle) {
        // ===== SAME VEHICLE: Model confused Level 1 ↔ Level 2 =====
        LOG_INFO("  -> Same vehicle (model confusion) - Kalman state preserved");
        
        // Transfer Kalman state from previous track to current main_CIPO
        for (auto& prev_obj : previous_objects_) {
            if (prev_obj.track_id == prev_main_cipo->track_id) {
                main_cipo.kalman = prev_obj.kalman;
                break;
            }
        }
    } else {
        // ===== DIFFERENT VEHICLE: Real cut-in! =====
        LOG_INFO("  -> Different vehicle (CUT-IN DETECTED) - Kalman filter reset");
        
        // Set event flags for visualization
        cut_in_detected_ = true;
        kalman_reset_ = true;
        
        // Reset Kalman filter for the NEW main_CIPO
        main_cipo.kalman.reset();
        main_cipo.kalman.initialize(main_cipo.distance_m);
        main_cipo.velocity_ms = 0.0f;  // Unknown velocity for cut-in vehicle
    }
    
    // ALWAYS copy velocity from main_cipo at the end
    cipo.velocity_ms = main_cipo.velocity_ms;
    
    return cipo;
}

TrackingResult ObjectFinder::updateAndGetCIPO(
    const std::vector<autospeed::Detection>& detections,
    const cv::Mat& frame)
{
    // Step 1: Perform tracking update (modifies internal tracked_objects_)
    update(detections, frame);
    
    // Step 2: Get CIPO (may modify main_CIPO's Kalman state in tracked_objects_)
    CIPOInfo cipo = getCIPO(frame);
    
    // Step 3: Package complete result with consistent state
    // All modifications to tracked_objects_ are done, so copy is consistent
    TrackingResult result;
    result.tracked_objects = tracked_objects_;  // Single copy after all modifications
    result.cipo = cipo;
    result.cut_in_detected = cut_in_detected_;
    result.kalman_reset = kalman_reset_;
    
    // Step 4: Clear event flags (ready for next frame)
    clearEventFlags();
    
    return result;
}

}  // namespace autoware_pov::vision::tracking
