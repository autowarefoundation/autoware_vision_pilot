#include "tracking/tracking_utils.hpp"
#include <algorithm>
#include <cmath>

namespace autoware_pov::vision::tracking {

float TrackingUtils::calculateIoU(const cv::Rect& a, const cv::Rect& b) {
    // Calculate intersection rectangle
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    
    // Calculate intersection area
    int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    
    // Calculate union area
    int union_area = a.area() + b.area() - intersection_area;
    
    // Return IoU
    return union_area > 0 ? static_cast<float>(intersection_area) / union_area : 0.0f;
}

float TrackingUtils::calculateCentroidDistance(const cv::Rect& a, const cv::Rect& b) {
    cv::Point2f centroid_a = getCentroid(a);
    cv::Point2f centroid_b = getCentroid(b);
    
    float dx = centroid_a.x - centroid_b.x;
    float dy = centroid_a.y - centroid_b.y;
    
    return std::sqrt(dx * dx + dy * dy);
}

float TrackingUtils::calculateSizeSimilarity(const cv::Rect& a, const cv::Rect& b) {
    float area_a = static_cast<float>(a.area());
    float area_b = static_cast<float>(b.area());
    
    if (area_a == 0.0f || area_b == 0.0f) {
        return 0.0f;
    }
    
    // Similarity = min(area_a, area_b) / max(area_a, area_b)
    // This gives 1.0 for identical sizes, approaching 0.0 for very different sizes
    float similarity = std::min(area_a, area_b) / std::max(area_a, area_b);
    
    return similarity;
}

cv::Point2f TrackingUtils::getCentroid(const cv::Rect& bbox) {
    return cv::Point2f(
        bbox.x + bbox.width / 2.0f,
        bbox.y + bbox.height / 2.0f
    );
}

cv::Point2f TrackingUtils::getBottomCenter(const cv::Rect& bbox) {
    return cv::Point2f(
        bbox.x + bbox.width / 2.0f,
        bbox.y + bbox.height  // Bottom edge
    );
}

float TrackingUtils::calculateMatchingScore(
    const cv::Rect& det_bbox,
    const cv::Rect& track_bbox,
    int image_width,
    int image_height
) {
    // Calculate individual metrics
    float iou = calculateIoU(det_bbox, track_bbox);
    float centroid_dist = calculateCentroidDistance(det_bbox, track_bbox);
    float size_similarity = calculateSizeSimilarity(det_bbox, track_bbox);
    
    // Normalize centroid distance to [0, 1] range
    // Maximum possible distance is diagonal of image
    float max_distance = std::sqrt(
        static_cast<float>(image_width * image_width + image_height * image_height)
    );
    float normalized_dist = 1.0f - std::min(centroid_dist / max_distance, 1.0f);
    
    // Weighted combination of metrics
    // IoU is most important (50%), then centroid proximity (30%), then size (20%)
    float matching_score = 0.5f * iou + 
                          0.3f * normalized_dist + 
                          0.2f * size_similarity;
    
    return matching_score;
}

}  // namespace autoware_pov::vision::tracking

