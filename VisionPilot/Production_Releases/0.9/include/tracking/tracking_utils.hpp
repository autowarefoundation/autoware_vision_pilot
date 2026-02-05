#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace autoware_pov::vision {

/**
 * @brief Utility functions for object tracking and data association
 * 
 * This module provides robust tracking utilities including:
 * - IoU (Intersection over Union) calculation
 * - Centroid distance measurement
 * - Bounding box similarity metrics
 */
class TrackingUtils {
public:
    /**
     * @brief Calculate Intersection over Union (IoU) between two bounding boxes
     * @param a First bounding box
     * @param b Second bounding box
     * @return IoU value in range [0.0, 1.0]
     */
    static float calculateIoU(const cv::Rect& a, const cv::Rect& b);
    
    /**
     * @brief Calculate Euclidean distance between centroids of two bounding boxes
     * @param a First bounding box
     * @param b Second bounding box
     * @return Distance in pixels
     */
    static float calculateCentroidDistance(const cv::Rect& a, const cv::Rect& b);
    
    /**
     * @brief Calculate size similarity ratio between two bounding boxes
     * @param a First bounding box
     * @param b Second bounding box
     * @return Similarity ratio in range [0.0, 1.0], where 1.0 = identical size
     */
    static float calculateSizeSimilarity(const cv::Rect& a, const cv::Rect& b);
    
    /**
     * @brief Get centroid point of a bounding box
     * @param bbox Bounding box
     * @return Centroid as cv::Point2f
     */
    static cv::Point2f getCentroid(const cv::Rect& bbox);
    
    /**
     * @brief Calculate bottom-center point of bbox (where object touches ground)
     * @param bbox Bounding box
     * @return Bottom-center point as cv::Point2f
     */
    static cv::Point2f getBottomCenter(const cv::Rect& bbox);
    
    /**
     * @brief Combined matching score for data association
     * 
     * Combines multiple cues (IoU, centroid distance, size similarity) into
     * a single matching score suitable for Hungarian algorithm or greedy matching.
     * 
     * @param det_bbox Detection bounding box
     * @param track_bbox Tracked object bounding box
     * @param image_width Image width for normalizing distances
     * @param image_height Image height for normalizing distances
     * @return Matching score in range [0.0, 1.0], higher is better
     */
    static float calculateMatchingScore(
        const cv::Rect& det_bbox,
        const cv::Rect& track_bbox,
        int image_width,
        int image_height
    );
};

}  // namespace autoware_pov::vision

