#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace autoware_pov::vision {

/**
 * @brief Feature extraction and matching utilities for object identity verification
 * 
 * Uses ORB (Oriented FAST and Rotated BRIEF) features for fast, robust matching.
 * Primary use: Verify if CIPO target changed to a different physical vehicle.
 */
class FeatureMatchingUtils {
public:
    /**
     * @brief Extract ORB features from an image region (bbox crop)
     * 
     * @param frame Full frame image
     * @param bbox Region of interest (bounding box)
     * @param keypoints Output: detected keypoints
     * @param descriptors Output: feature descriptors
     * @return true if features extracted successfully, false if too few features
     */
    static bool extractFeatures(const cv::Mat& frame,
                                const cv::Rect& bbox,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors);
    
    /**
     * @brief Match features between two sets of descriptors
     * 
     * Uses Hamming distance with Lowe's ratio test for robust matching.
     * 
     * @param descriptors1 First set of descriptors
     * @param descriptors2 Second set of descriptors
     * @param good_matches Output: filtered good matches
     * @return Number of good matches found
     */
    static int matchFeatures(const cv::Mat& descriptors1,
                            const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& good_matches);
    
    /**
     * @brief Calculate match confidence (similarity) between two objects
     * 
     * Computes a normalized confidence score based on:
     * - Number of good matches
     * - Number of keypoints in both images
     * 
     * @param descriptors1 First object descriptors
     * @param descriptors2 Second object descriptors
     * @param num_keypoints1 Number of keypoints in first object
     * @param num_keypoints2 Number of keypoints in second object
     * @return Confidence in range [0.0, 1.0], higher = more similar
     */
    static float calculateMatchConfidence(const cv::Mat& descriptors1,
                                         const cv::Mat& descriptors2,
                                         int num_keypoints1,
                                         int num_keypoints2);
    
    /**
     * @brief Extract a safe crop from frame (handles boundary cases)
     * 
     * @param frame Source frame
     * @param bbox Desired bounding box
     * @return Cropped image (may be smaller if bbox extends outside frame)
     */
    static cv::Mat extractSafeCrop(const cv::Mat& frame, const cv::Rect& bbox);
    
    /**
     * @brief Determine if two objects are the same based on feature matching
     * 
     * High-level function that combines extraction, matching, and thresholding.
     * 
     * @param frame1 First frame
     * @param bbox1 First object bbox
     * @param frame2 Second frame  
     * @param bbox2 Second object bbox
     * @param confidence_threshold Minimum confidence to consider same object (default: 0.3)
     * @return true if objects are likely the same, false otherwise
     */
    static bool areSameObject(const cv::Mat& frame1,
                             const cv::Rect& bbox1,
                             const cv::Mat& frame2,
                             const cv::Rect& bbox2,
                             float confidence_threshold = 0.3f);

private:
    // ORB feature detector (static to avoid recreation)
    static cv::Ptr<cv::ORB> getORBDetector();
    
    // Brute-force matcher for ORB features
    static cv::Ptr<cv::BFMatcher> getBFMatcher();
    
    // Lowe's ratio test threshold for filtering matches
    static constexpr float LOWE_RATIO = 0.75f;
};

}  // namespace autoware_pov::vision

