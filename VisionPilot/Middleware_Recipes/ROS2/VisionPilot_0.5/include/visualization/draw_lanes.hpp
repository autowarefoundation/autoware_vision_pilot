#ifndef AUTOWARE_POV_VISION_EGOLANES_DRAW_LANES_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_DRAW_LANES_HPP_

#include "../inference/lane_segmentation.hpp"
#include "../lane_tracking/lane_tracking.hpp"
#include <opencv2/opencv.hpp>
#include <optional>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief Visualize lane segmentation on input image
 * 
 * Draws colored circles on detected lane pixels:
 * - Blue: Ego left lane
 * - Magenta: Ego right lane  
 * - Green: Other lanes
 * 
 * @param input_image Original input image (any resolution)
 * @param lanes Lane segmentation masks (typically 320x640)
 * @return Annotated image (same size as input)
 */
cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes
);

/**
 * @brief In-place lane visualization (modifies input image)
 * 
 * @param image Image to annotate (modified in-place)
 * @param lanes Lane segmentation masks
 */
void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
);

/**
 * @brief In-place visualization of filtered lanes
 * 
 * @param image Image to annotate (modified in-place)
 * @param lanes Filtered lane segmentation masks
 */
void drawFilteredLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

/**
 * @brief Draws ONLY the raw 160x80 pixel masks overlay.
 * Useful for debugging the model output and RANSAC inputs.
 * Later on I might add some more, like sliding windows too etc.
 */
void drawRawMasksInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
);

/**
 * @brief Draws ONLY the smooth polynomial fitted lines.
 * This represents the final product output Insha'Allah.
 */
void drawPolyFitLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
);

/**
 * @brief Draws BEV vis panel.
 * * @param image Output img to draw on (will be resized to 640x640).
 * @param orig_frame Orig perspective frame.
 * @param bev_data BEV vis data from the tracker.
 */
void drawBEVVis(
  cv::Mat& image,
  const cv::Mat& orig_frame,
  const BEVVisuals& bev_data
);

/**
 * @brief Debug function to verify Pixel -> Meter conversion
 * Draws the metric polynomials (projected back to pixels) on top of the BEV image.
 */
void drawMetricVerification(
    cv::Mat& bev_image,
    const std::vector<double>& left_metric_coeffs,
    const std::vector<double>& right_metric_coeffs
);

cv::Mat rotateSteeringWheel(const cv::Mat& img, float steering_angle_deg);
void visualizeWheel(const cv::Mat& img, const cv::Mat& wheelImg, const int x, const int y);
void visualizeSteering(
  cv::Mat& img,
  float steering_angle,
  const cv::Mat& rotatedPredSteeringWheelImg,
  std::optional<float> gtSteeringAngle,
  const cv::Mat& rotatedGtSteeringWheelImg);

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_DRAW_LANES_HPP_