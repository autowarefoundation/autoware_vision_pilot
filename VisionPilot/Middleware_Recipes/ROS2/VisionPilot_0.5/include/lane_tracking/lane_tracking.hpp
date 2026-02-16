#ifndef AUTOWARE_POV_VISION_EGOLANES_LANE_TRACKING_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_LANE_TRACKING_HPP_

#include "inference/lane_segmentation.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace autoware_pov::vision::egolanes
{

// BEV-specific visualization data
struct BEVVisuals {
    cv::Mat H_orig_to_bev;                // Homography for warping frame
    std::vector<double> bev_left_coeffs;  // BEV egoleft coeffs
    std::vector<double> bev_right_coeffs; // BEV egoright coeffs
    std::vector<double> bev_center_coeffs;// BEV drivable corridor coeffs
    
    // BEV lane points in pixel space (for PathFinder)
    std::vector<cv::Point2f> bev_left_pts;   // Left lane points in BEV pixels
    std::vector<cv::Point2f> bev_right_pts;  // Right lane points in BEV pixels
    
    double last_valid_width_pixels = 0.0; // Width bar
    bool valid = false;
};

/**
 * @brief Container for curve params both views
 * (for statistics and debugging later)
 */
struct DualViewMetrics {

    // Original perspective curve params
    double orig_lane_offset = 0.0;
    double orig_yaw_offset = 0.0;
    double orig_curvature = 0.0;

    // BEV perspective curve params
    double bev_lane_offset = 0.0;
    double bev_yaw_offset = 0.0;
    double bev_curvature = 0.0;

    // BEV visuals metrics
    BEVVisuals bev_visuals;
};

class LaneTracker {

public:

    LaneTracker();
    ~LaneTracker() = default;

    /**
     * @brief Main processing func
     * 1. Warp valid egoline to BEV
     * 2. Recover missing egoline in BEV via last-known-good-frame's lane width shifting
     * 3. Calc curve params in BEV
     * 4. Warp back to perspective to update vis
     * * @param input_lanes Input from LaneFilter
     * @param image_size Size of full input img
     * @return Updated LaneSegmentation with recovered egolines and 6 curve params (3 for each view)
     */
    std::pair<LaneSegmentation, DualViewMetrics> update(
        const LaneSegmentation& input_lanes,
        const cv::Size& image_size
    );

private:

    // STATE PARAMS

    // Homomatrix, hard-coded from calibration
    cv::Mat H_orig_to_bev = (cv::Mat_<double>(3,3) <<
        -1.79887412e-01, -6.05811422e-01,  6.02998251e+02,
        1.85824549e-14 , -1.28170839e+00,  8.63871455e+02,
        2.95628463e-17 , -1.76125061e-03,  1.00000000e+00
    );

    // Inversed homomatrix
    cv::Mat H_bev_to_orig = H_orig_to_bev.inv();
    
    bool homography_inited = false;
    cv::Size cached_image_size;

    // BEV lane width cache (in BEV pixels)
    // Updated whenever there are 2 valid egolines.
    // If one missing, use this to shift the available one.
    // If both lost, fuck it I'm out.
    double last_valid_bev_width = 180.0;    // Default fallback (tuned for 640x640 BEV)
    bool has_valid_width_history = false;

    // BEV HELPERS

    // Init homography
    void initHomography(const cv::Size& image_size);

    // Coords transforms
    std::vector<cv::Point2f> warpPoints(
        const std::vector<cv::Point2f>& src_pts, 
        const cv::Mat& H
    );

    // Polynomial
    std::vector<cv::Point2f> genPointsFromCoeffs(
        const std::vector<double>& coeffs, 
        int height,
        int step = 5
    );

    std::vector<double> fitPoly2ndOrder(
        const std::vector<cv::Point2f>& points,
        int img_height
    );

    // MATH HELPERS
    double calcCurvature(
        const std::vector<double>& coeffs, 
        double y_eval
    );
    double calcYawOffset(
        const std::vector<double>& coeffs, 
        double y_eval
    );
    double calcLaneOffset(
        const std::vector<double>& coeffs, 
        double y_eval
    );

};

} // namespace autoware_pov::vision::egolanes

#endif // AUTOWARE_POV_VISION_EGOLANES_LANE_TRACKING_HPP_