/**
 * @file visualize.hpp
 * @brief Visualization functions for EgoLanes lateral control
 */

#ifndef AUTOWARE_POV_VISION_EGOLANES_VISUALIZE_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_VISUALIZE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include "inference/lane_segmentation.hpp"
#include "lane_tracking/lane_tracking.hpp"
#include "path_planning/path_finder.hpp"

namespace autoware_pov::vision::egolanes
{

/**
 * @brief Draw lane segmentation on image (returns new copy)
 * @param input_image Input frame
 * @param lanes Lane segmentation masks
 * @return Visualization image with lanes overlaid
 */
cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes);

/**
 * @brief Draw lane segmentation on image (in-place)
 * @param image Input/output frame
 * @param lanes Lane segmentation masks
 */
void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

/**
 * @brief Draw filtered lanes (debug visualization)
 * @param image Input/output frame
 * @param lanes Filtered lane segmentation
 */
void drawFilteredLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

/**
 * @brief Draw raw segmentation masks (red/green/blue channels)
 * @param image Input/output frame
 * @param lanes Lane segmentation masks
 */
void drawRawMasksInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

/**
 * @brief Draw polynomial-fitted lanes
 * @param image Input/output frame
 * @param lanes Lane segmentation with polynomial coefficients
 */
void drawPolyFitLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

/**
 * @brief Draw BEV (Bird's Eye View) visualization
 * @param bev_canvas Output BEV canvas (640x640)
 * @param original_frame Original input frame
 * @param bev_visuals BEV visualization data from LaneTracker
 */
void drawBEVVis(
  cv::Mat& bev_canvas,
  const cv::Mat& original_frame,
  const BEVVisuals& bev_visuals);

/**
 * @brief Draw metric verification (projected back from metric space)
 * @param bev_canvas BEV canvas to draw on
 * @param left_coeffs Left lane polynomial coefficients (metric space)
 * @param right_coeffs Right lane polynomial coefficients (metric space)
 */
void drawMetricVerification(
  cv::Mat& bev_canvas,
  const std::vector<double>& left_coeffs,
  const std::vector<double>& right_coeffs);

/**
 * @brief Rotate steering wheel image
 * @param img Steering wheel image
 * @param steering_angle_deg Steering angle in degrees
 * @return Rotated steering wheel image
 */
cv::Mat rotateSteeringWheel(const cv::Mat& img, float steering_angle_deg);

/**
 * @brief Visualize steering wheel on image
 * @param img Input frame
 * @param wheelImg Steering wheel image (pre-rotated)
 * @param x X position
 * @param y Y position
 */
void visualizeWheel(const cv::Mat& img, const cv::Mat& wheelImg, const int x, const int y);

/**
 * @brief Visualize steering angles (predicted and ground truth)
 * @param img Input/output frame
 * @param steering_angle Predicted steering angle (degrees)
 * @param rotatedPredSteeringWheelImg Predicted steering wheel image (rotated)
 * @param gtSteeringAngle Ground truth steering angle (optional)
 * @param rotatedGtSteeringWheelImg Ground truth steering wheel image (rotated)
 */
void visualizeSteering(
  cv::Mat& img,
  const float steering_angle,
  const cv::Mat& rotatedPredSteeringWheelImg,
  std::optional<float> gtSteeringAngle,
  const cv::Mat& rotatedGtSteeringWheelImg);

/**
 * @brief Show lane departure warning on image
 * @param img Input/output frame
 */
void showLaneDepartureWarning(cv::Mat& img);

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_VISUALIZE_HPP_

