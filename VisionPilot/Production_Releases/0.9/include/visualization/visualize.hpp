#ifndef VISUALIZATION_VISUALIZE_HPP_
#define VISUALIZATION_VISUALIZE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include "lane_tracking/lane_tracking.hpp"

namespace autoware_pov::vision::egolanes {

// Forward declarations
struct LaneSegmentation;
struct BEVVisuals;

// Draw lanes on a copy of the input image
cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes);

// Draw lanes in-place on the input image
void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

// Draw filtered lanes with polynomial fits
void drawFilteredLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

// Draw raw network output masks
void drawRawMasksInPlace(
    cv::Mat& image,
    const LaneSegmentation& lanes);

// Draw polynomial fitted lanes
void drawPolyFitLanesInPlace(
    cv::Mat& image,
    const LaneSegmentation& lanes);

// Draw BEV (Bird's Eye View) visualization
void drawBEVVis(
    cv::Mat& image,
    const cv::Mat& orig_frame,
    const BEVVisuals& bev_data);

// Draw metric verification overlay
void drawMetricVerification(
    cv::Mat& bev_image,
    const std::vector<double>& left_metric_coeffs,
    const std::vector<double>& right_metric_coeffs);

// Rotate steering wheel image by angle
cv::Mat rotateSteeringWheel(const cv::Mat& img, float steering_angle_deg);

// Visualize steering wheel on image
void visualizeWheel(const cv::Mat& img, const cv::Mat& wheelImg, const int x, const int y);

// Visualize steering with prediction and optional ground truth
void visualizeSteering(
  cv::Mat& img,
  const float steering_angle,
  const cv::Mat& rotatedPredSteeringWheelImg,
  std::optional<float> gtSteeringAngle,
  const cv::Mat& rotatedGtSteeringWheelImg);

// Show lane departure warning overlay
void showLaneDepartureWarning(cv::Mat& img);

} // namespace autoware_pov::vision::egolanes

#endif // VISUALIZATION_VISUALIZE_HPP_

