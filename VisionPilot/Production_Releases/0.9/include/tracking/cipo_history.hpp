#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <chrono>

namespace autoware_pov::vision {

/**
 * @brief Stores information about CIPO at a specific frame
 */
struct CIPOSnapshot {
    int track_id;                                       // Tracking ID of the CIPO
    int class_id;                                       // Class ID (1 or 2)
    cv::Rect bbox;                                      // Bounding box
    float distance_m;                                   // Distance in meters
    float velocity_ms;                                  // Velocity in m/s
    std::chrono::steady_clock::time_point timestamp;   // Frame timestamp
    
    // Frame crop for feature matching
    cv::Mat frame_crop;                                 // Image region of the CIPO bbox
};

/**
 * @brief Manages historical CIPO data using a circular buffer
 * 
 * This class maintains a fixed-size history of CIPO observations across frames.
 * When the buffer is full, oldest entries are automatically removed (FIFO).
 * 
 * Primary uses:
 * 1. Detect CIPO target changes (different track_id)
 * 2. Future: Store visual features for identity verification
 * 3. Analyze CIPO behavior patterns (distance trends, velocity smoothness)
 */
class CIPOHistory {
public:
    /**
     * @brief Construct CIPO history manager
     * @param max_history_size Maximum number of frames to store (default: 30)
     */
    explicit CIPOHistory(size_t max_history_size = 30);
    
    /**
     * @brief Add a new CIPO observation
     * @param snapshot CIPO snapshot for current frame
     */
    void push(const CIPOSnapshot& snapshot);
    
    /**
     * @brief Get the most recent CIPO snapshot
     * @return Pointer to latest snapshot, or nullptr if history is empty
     */
    const CIPOSnapshot* getLatest() const;
    
    /**
     * @brief Get the previous CIPO snapshot (one frame before latest)
     * @return Pointer to previous snapshot, or nullptr if not available
     */
    const CIPOSnapshot* getPrevious() const;
    
    /**
     * @brief Check if CIPO target changed between last two frames
     * @return true if track_id changed, false otherwise
     */
    bool didCIPOChange() const;
    
    /**
     * @brief Get the full history buffer (read-only)
     * @return Reference to the deque containing all snapshots
     */
    const std::deque<CIPOSnapshot>& getHistory() const { return history_; }
    
    /**
     * @brief Get number of stored snapshots
     * @return Size of history buffer
     */
    size_t size() const { return history_.size(); }
    
    /**
     * @brief Clear all history
     */
    void clear();
    
    /**
     * @brief Check if history is empty
     * @return true if no snapshots stored
     */
    bool empty() const { return history_.empty(); }

private:
    std::deque<CIPOSnapshot> history_;  // Circular buffer for CIPO history
    size_t max_size_;                   // Maximum buffer size
};

}  // namespace autoware_pov::vision

