#include "../../include/tracking/feature_matching_utils.hpp"
#include "../../include/tracking/logging.hpp"
#include <algorithm>

namespace autoware_pov::vision {

cv::Ptr<cv::ORB> FeatureMatchingUtils::getORBDetector() {
    // Static detector - created once and reused
    static cv::Ptr<cv::ORB> orb = cv::ORB::create(
        500,        // nfeatures: max number of features to retain
        1.2f,       // scaleFactor: pyramid decimation ratio
        8,          // nlevels: number of pyramid levels
        31,         // edgeThreshold: size of border where features not detected
        0,          // firstLevel: level of pyramid to put source image
        2,          // WTA_K: number of points for oriented BRIEF descriptor
        cv::ORB::HARRIS_SCORE,  // scoreType: HARRIS_SCORE or FAST_SCORE
        31,         // patchSize: size of patch used by oriented BRIEF
        20          // fastThreshold: fast threshold
    );
    return orb;
}

cv::Ptr<cv::BFMatcher> FeatureMatchingUtils::getBFMatcher() {
    // Static matcher - uses Hamming distance for ORB descriptors
    static cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    return matcher;
}

cv::Mat FeatureMatchingUtils::extractSafeCrop(const cv::Mat& frame, const cv::Rect& bbox) {
    // Ensure bbox is within frame boundaries
    cv::Rect safe_bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    
    if (safe_bbox.area() == 0) {
        return cv::Mat();  // Return empty if no overlap
    }
    
    return frame(safe_bbox).clone();
}

bool FeatureMatchingUtils::extractFeatures(const cv::Mat& frame,
                                           const cv::Rect& bbox,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors) {
    // Extract crop from frame
    cv::Mat crop = extractSafeCrop(frame, bbox);
    
    if (crop.empty() || crop.cols < 20 || crop.rows < 20) {
        return false;  // Crop too small
    }
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (crop.channels() == 3) {
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = crop;
    }
    
    // Detect and compute ORB features
    cv::Ptr<cv::ORB> orb = getORBDetector();
    keypoints.clear();
    descriptors = cv::Mat();
    
    orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    // Require minimum number of features
    const int MIN_FEATURES = 10;
    if (keypoints.size() < MIN_FEATURES) {
        return false;
    }
    
    return true;
}

int FeatureMatchingUtils::matchFeatures(const cv::Mat& descriptors1,
                                       const cv::Mat& descriptors2,
                                       std::vector<cv::DMatch>& good_matches) {
    good_matches.clear();
    
    if (descriptors1.empty() || descriptors2.empty()) {
        return 0;
    }
    
    // Perform k-nearest-neighbors matching (k=2 for ratio test)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::Ptr<cv::BFMatcher> matcher = getBFMatcher();
    
    try {
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    } catch (const cv::Exception& e) {
        LOG_INFO(("Feature matching exception: " + std::string(e.what())).c_str());
        return 0;
    }
    
    // Apply Lowe's ratio test to filter good matches
    for (const auto& knn_match : knn_matches) {
        if (knn_match.size() >= 2) {
            if (knn_match[0].distance < LOWE_RATIO * knn_match[1].distance) {
                good_matches.push_back(knn_match[0]);
            }
        }
    }
    
    return good_matches.size();
}

float FeatureMatchingUtils::calculateMatchConfidence(const cv::Mat& descriptors1,
                                                    const cv::Mat& descriptors2,
                                                    int num_keypoints1,
                                                    int num_keypoints2) {
    std::vector<cv::DMatch> good_matches;
    int num_matches = matchFeatures(descriptors1, descriptors2, good_matches);
    
    if (num_matches == 0 || num_keypoints1 == 0 || num_keypoints2 == 0) {
        return 0.0f;
    }
    
    // Confidence = number of matches / average number of keypoints
    // This normalizes for different numbers of features
    float avg_keypoints = (num_keypoints1 + num_keypoints2) / 2.0f;
    float confidence = std::min(num_matches / avg_keypoints, 1.0f);
    
    return confidence;
}

bool FeatureMatchingUtils::areSameObject(const cv::Mat& frame1,
                                        const cv::Rect& bbox1,
                                        const cv::Mat& frame2,
                                        const cv::Rect& bbox2,
                                        float confidence_threshold) {
    // Extract features from both objects
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    bool success1 = extractFeatures(frame1, bbox1, keypoints1, descriptors1);
    bool success2 = extractFeatures(frame2, bbox2, keypoints2, descriptors2);
    
    if (!success1 || !success2) {
        LOG_INFO("Feature extraction failed for one or both objects");
        return false;  // Can't determine, assume different
    }
    
    // Calculate match confidence
    float confidence = calculateMatchConfidence(
        descriptors1, descriptors2,
        keypoints1.size(), keypoints2.size()
    );
    
    LOG_INFO(("Feature match confidence: " + std::to_string(confidence) + 
              " (threshold: " + std::to_string(confidence_threshold) + ")").c_str());
    
    return confidence >= confidence_threshold;
}

}  // namespace autoware_pov::vision

