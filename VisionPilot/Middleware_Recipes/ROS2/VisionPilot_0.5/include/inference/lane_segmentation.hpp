#ifndef AUTOWARE_POV_VISION_EGOLANES_LANE_SEGMENTATION_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_LANE_SEGMENTATION_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief Lane segmentation output structure
 * 
 * Contains binary segmentation masks for different lane types.
 * Each mask is a 2D binary image (1.0 = lane present, 0.0 = no lane)
 */
struct LaneSegmentation
{
  cv::Mat ego_left;    // Ego left lane boundary (320x640)
  cv::Mat ego_right;   // Ego right lane boundary (320x640)
  cv::Mat other_lanes; // Other visible lanes (320x640)
  
  int height;
  int width;

  // Coeffs for polyfit upscaling
  std::vector<double> left_coeffs;
  std::vector<double> right_coeffs;

  // Drivable path info
  std::vector<double> center_coeffs;

  // Curve params info
  double lane_offset = 0.0;     // Distance from image center to path center
  double yaw_offset = 0.0;      // Heading error (radians)
  double steering_angle = 0.0;  // Steering angle (degrees)
  double curvature = 0.0;       // Curveture = 1/R
  bool path_valid = false;      // True if we successfully calculated the path

  // Debug info
  cv::Point left_start_point = {-1, -1};
  cv::Point right_start_point = {-1, -1};
  std::vector<cv::Rect> left_sliding_windows;
  std::vector<cv::Rect> right_sliding_windows;
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_LANE_SEGMENTATION_HPP_

