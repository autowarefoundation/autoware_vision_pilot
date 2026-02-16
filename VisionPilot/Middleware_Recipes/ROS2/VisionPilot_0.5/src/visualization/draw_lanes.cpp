#include "visualization/draw_lanes.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

namespace autoware_pov::vision::egolanes
{

cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes)
{
  // Clone input for visualization
  cv::Mat vis_image = input_image.clone();
  drawLanesInPlace(vis_image, lanes);
  return vis_image;
}

void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes)
{
  // Calculate scale from lane mask to input image
  float scale_x = static_cast<float>(image.cols) / lanes.width;
  float scale_y = static_cast<float>(image.rows) / lanes.height;

  // Choose radius based on scale
  float base_scale = std::min(scale_x, scale_y);
  int radius_ = std::max(1, static_cast<int>(std::round(base_scale * 0.5f)));

  // Define colors (BGR format for OpenCV)
  cv::Scalar color_ego_left(255, 0, 0);      // Blue
  cv::Scalar color_ego_right(255, 0, 200);   // Magenta
  cv::Scalar color_other(0, 153, 0);         // Green

  // Draw ego left lane
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.ego_left.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y),
                   radius_, color_ego_left, -1, cv::LINE_AA);
      }
    }
  }

  // Draw ego right lane
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.ego_right.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y),
                   radius_, color_ego_right, -1, cv::LINE_AA);
      }
    }
  }

  // Draw other lanes
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.other_lanes.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y),
                   radius_, color_other, -1, cv::LINE_AA);
      }
    }
  }
}

// Helper func to gen points from polynimial with scaling
static std::vector<cv::Point> genSmoothCurve(
    const std::vector<double>& coeffs,
    int img_width,
    int img_height,
    int model_width,
    int model_height
)
{
    std::vector<cv::Point> points;
    if (coeffs.size() < 6) return points;

    double a         = coeffs[0];
    double b         = coeffs[1];
    double c         = coeffs[2];
    double d         = coeffs[3];
    double min_y_lim = coeffs[4];
    double max_y_lim = coeffs[5];

    // Scaling factors
    // Polyfit coeffs are calculated in model space (160x80)
    // But gotta bring em into image space (640x360 or smth)
    double scale_y = static_cast<double>(model_height) / img_height;
    double scale_x = static_cast<double>(img_width) / model_width;

    // Lims in image space
    int img_y_start = static_cast<int>(min_y_lim / scale_y);
    int img_y_end   = static_cast<int>(max_y_lim / scale_y);
    img_y_start = std::max(
      0,
      img_y_start
    );
    img_y_end   = std::min(
      img_height - 1,
      img_y_end
    );

    // Iterate ONLY within valid Y-range
    for (int y_img = img_y_start; y_img <= img_y_end; ++y_img) {

        // 1. Image Y -> model Y
        double y_model = y_img * scale_y;

        // 2. Calc X in model space via poly
        double x_model = a * std::pow(y_model, 3) +
                         b * std::pow(y_model, 2) +
                         c * y_model +
                         d;

        // 3. Model X -> image X
        double x_img = x_model * scale_x;

        // 4. Store valid points
        if (x_img >= 0 && x_img < img_width) {
            points.push_back(
              cv::Point(
                static_cast<int>(x_img),
                y_img
              )
            );
        }
    }
    return points;
}

void drawFilteredLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
)
{

    // Define colors (BGR format for OpenCV)
    cv::Scalar color_ego_left(255, 0, 0);      // Blue
    cv::Scalar color_ego_right(255, 0, 200);   // Magenta
    cv::Scalar color_other(0, 153, 0);         // Green

    // 1a. Raw mask of other lines
    if (!lanes.other_lanes.empty()) {
        cv::Mat other_mask_resized;
        // Nearest neighbor resize to keep it binary-ish before smoothing
        cv::resize(
          lanes.other_lanes,
          other_mask_resized,
          image.size(),
          0, 0,
          cv::INTER_NEAREST
        );

        // Create green overlay
        cv::Mat green_layer(
          image.size(),
          image.type(),
          color_other
        );

        cv::Mat mask_8u;
        other_mask_resized.convertTo(
          mask_8u,
          CV_8U,
          255.0
        );

        // Apply simple threshold to clean up
        cv::threshold(
          mask_8u,
          mask_8u,
          127,
          255,
          cv::THRESH_BINARY
        );

        cv::Mat overlay;
        green_layer.copyTo(
          overlay,
          mask_8u
        );

        // Add faint green glow
        cv::addWeighted(
          image,
          1.0,
          overlay,
          0.4,
          0,
          image
        );
    }

    // 1b. Raw mask of ego left line
    if (!lanes.ego_left.empty()) {
        cv::Mat ego_left_resized;
        cv::resize(
          lanes.ego_left,
          ego_left_resized,
          image.size(),
          0, 0,
          cv::INTER_NEAREST
        );

        cv::Mat mask_left_8u;
        ego_left_resized.convertTo(
          mask_left_8u,
          CV_8U,
          255.0
        );

        cv::threshold(
          mask_left_8u,
          mask_left_8u,
          127,
          255,
          cv::THRESH_BINARY
        );

        cv::Mat blue_layer(image.size(), image.type(), color_ego_left);
        cv::Mat overlay_left;
        blue_layer.copyTo(overlay_left, mask_left_8u);
        cv::addWeighted(image, 1.0, overlay_left, 0.35, 0, image);
    }

    // 1c. Raw mask of ego right line
    if (!lanes.ego_right.empty()) {
        cv::Mat ego_right_resized;
        cv::resize(
          lanes.ego_right,
          ego_right_resized,
          image.size(),
          0, 0,
          cv::INTER_NEAREST
        );

        cv::Mat mask_right_8u;
        ego_right_resized.convertTo(
          mask_right_8u,
          CV_8U,
          255.0
        );

        cv::threshold(
          mask_right_8u,
          mask_right_8u,
          127,
          255,
          cv::THRESH_BINARY
        );

        cv::Mat magenta_layer(image.size(), image.type(), color_ego_right);
        cv::Mat overlay_right;
        magenta_layer.copyTo(overlay_right, mask_right_8u);
        cv::addWeighted(image, 1.0, overlay_right, 0.35, 0, image);
    }

    // 2. EgoLeft
    if (!lanes.left_coeffs.empty()) {
        auto left_points = genSmoothCurve(
            lanes.left_coeffs,
            image.cols,
            image.rows,
            lanes.width,
            lanes.height
        );
        // Polyline
        cv::polylines(
          image,
          left_points,
          false,
          color_ego_left,
          5,
          cv::LINE_AA
        );
    }

    // 3. EgoRight
    if (!lanes.right_coeffs.empty()) {
        auto right_points = genSmoothCurve(
            lanes.right_coeffs,
            image.cols,
            image.rows,
            lanes.width,
            lanes.height
        );
        // Polyline
        cv::polylines(
          image,
          right_points,
          false,
          color_ego_right,
          5,
          cv::LINE_AA
        );
      }
}

// ========================== MAIN VIS VIEWS - DEBUGGING + FINAL OUTPUTS ========================== //

// Helper func: draw mask overlay only
static void drawMaskOverlay(
  cv::Mat& image,
  const cv::Mat& mask,
  const cv::Scalar& color
)
{
    if (mask.empty()) return;

    cv::Mat mask_resized;
    cv::resize(
      mask,
      mask_resized,
      image.size(),
      0,
      0,
      cv::INTER_NEAREST
    );

    cv::Mat color_layer(
      image.size(),
      image.type(),
      color
    );
    cv::Mat mask_8u;
    mask_resized.convertTo(
      mask_8u,
      CV_8U,
      255.0
    );
    cv::threshold(
      mask_8u,
      mask_8u,
      127,
      255,
      cv::THRESH_BINARY
    );

    cv::Mat overlay;
    color_layer.copyTo(
      overlay,
      mask_8u
    );
    cv::addWeighted(
      image,
      1.0,
      overlay,
      0.4,
      0,
      image
    );
}

// Helper func: draw raw masks (debug view)
void drawRawMasksInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
)
{
    cv::Scalar color_ego_left(255, 0, 0);      // Blue
    cv::Scalar color_ego_right(255, 0, 200);   // Magenta
    cv::Scalar color_other(0, 153, 0);         // Green

    drawMaskOverlay(
      image,
      lanes.other_lanes,
      color_other
    );
    drawMaskOverlay(
      image,
      lanes.ego_left,
      color_ego_left
    );
    drawMaskOverlay(
      image,
      lanes.ego_right,
      color_ego_right
    );

    // // Draw sliding windows for debugging
    // float scale_x = static_cast<float>(image.cols) / lanes.width;
    // float scale_y = static_cast<float>(image.rows) / lanes.height;
    // cv::Scalar color_window(0, 0, 255); // Red

    // // a. Left windows
    // for (const auto& rect : lanes.left_sliding_windows) {
    //     cv::Rect scaled_rect(
    //         static_cast<int>(rect.x * scale_x),
    //         static_cast<int>(rect.y * scale_y),
    //         static_cast<int>(rect.width * scale_x),
    //         static_cast<int>(rect.height * scale_y)
    //     );
    //     cv::rectangle(image, scaled_rect, color_window, 1);
    // }

    // // b. Right windows
    // for (const auto& rect : lanes.right_sliding_windows) {
    //     cv::Rect scaled_rect(
    //         static_cast<int>(rect.x * scale_x),
    //         static_cast<int>(rect.y * scale_y),
    //         static_cast<int>(rect.width * scale_x),
    //         static_cast<int>(rect.height * scale_y)
    //     );
    //     cv::rectangle(image, scaled_rect, color_window, 1);
    // }

    // cv::putText(
    //   image,
    //   "DEBUG: Raw masks",
    //   cv::Point(20, 40),
    //   cv::FONT_HERSHEY_SIMPLEX,
    //   1.0,
    //   cv::Scalar(0, 255, 255),
    //   2
    // );
}

// Helper func: draw smooth polyfit lines only (final prod view)
void drawPolyFitLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
)
{
  cv::Scalar color_ego_left(255, 0, 0);     // Blue
  cv::Scalar color_ego_right(255, 0, 200);  // Magenta
  cv::Scalar color_center(0, 255, 255);     // Yellow

  // Draw vectors

  // Egoleft
  if (!lanes.left_coeffs.empty()) {
    auto left_points = genSmoothCurve(
      lanes.left_coeffs,
      image.cols,
      image.rows,
      lanes.width,
      lanes.height
    );
    if (left_points.size() > 1) {
      cv::polylines(
        image,
        left_points,
        false,
        color_ego_left,
        15,
        cv::LINE_AA
      );
      cv::polylines(
        image,
        left_points,
        false,
        cv::Scalar(255, 200, 0),
        5,
        cv::LINE_AA
      );
    }
  }

  // Egoright
  if (!lanes.right_coeffs.empty()) {
    auto right_points = genSmoothCurve(
      lanes.right_coeffs,
      image.cols,
      image.rows,
      lanes.width,
      lanes.height
    );
    if (right_points.size() > 1) {
      cv::polylines(
        image,
        right_points,
        false,
        color_ego_right,
        15,
        cv::LINE_AA
      );
      cv::polylines(
        image,
        right_points,
        false,
        cv::Scalar(255, 150, 255),
        5,
        cv::LINE_AA
      );
    }
  }

  // Drivable path
  if (
    lanes.path_valid &&
    !lanes.center_coeffs.empty()
  )
  {
    std::vector<double> viz_coeffs = lanes.center_coeffs;
    viz_coeffs[5] = static_cast<double>(lanes.height - 1);  // Extend to bottom

    auto center_points = genSmoothCurve(
      viz_coeffs,
      image.cols,
      image.rows,
      lanes.width,
      lanes.height
    );

    if (center_points.size() > 1) {
      cv::polylines(
        image,
        center_points,
        false,
        color_center,
        15,
        cv::LINE_AA
      );
      cv::polylines(
        image,
        center_points,
        false,
        cv::Scalar(255, 255, 255),
        5,
        cv::LINE_AA
      );
    }

    // Params info as text for now
    std::vector<std::string> lines;
    lines.push_back(cv::format("Lane offset: %.2f px", lanes.lane_offset));
    lines.push_back(cv::format("Yaw offset: %.2f rad", lanes.yaw_offset));
    // lines.push_back(cv::format("Steering angle: %.2f deg", lanes.steering_angle));
    lines.push_back(cv::format("Curvature: %.4f", lanes.curvature));

    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 1.2;
    int thickness = 2;
    int line_spacing = 10; // extra spacing
    int margin = 50;
    int y = margin;

    for (const auto& l : lines) {
      cv::Size textSize = cv::getTextSize(
        l,
        font,
        scale,
        thickness,
        nullptr
      );
      int x = image.cols - textSize.width - margin;  // Align right
      cv::putText(
        image,
        l,
        cv::Point(x, y),
        font,
        scale,
        color_center,
        thickness
      );
      y += textSize.height + line_spacing;
    }
  }

  cv::putText(
    image,
    "FINAL: RANSAC polyfit",
    cv::Point(20, 40),
    cv::FONT_HERSHEY_SIMPLEX,
    1.0,
    cv::Scalar(0, 255, 0),
    2
  );
}

// ========================== ADDITIONAL VIS VIEW - BEV ========================== //

// Helper func: gen points from coeffs directly in BEV space (no scaling needed)
static std::vector<cv::Point> genBEVPoints(
  const std::vector<double>& coeffs,
  int bev_height = 640
)
{
  std::vector<cv::Point> points;
  // Now using quadratic coeffs: [0, a, b, c, min_y, max_y]
  if (coeffs.size() < 6) return points;

  double a = coeffs[1];
  double b = coeffs[2];
  double c = coeffs[3];
  double min_y = coeffs[4];
  double max_y = coeffs[5];

  for (int y = 0; y < bev_height; ++y) {
    // Only draw within valid y-range defined by fitted points
    if (y < min_y || y > max_y) continue;

    // x = ay^2 + by + c
    double x = a*y*y + b*y + c;

    // BEV grid is 640 wide
    if (x >= 0 && x < 640) {
      points.push_back(cv::Point(
        static_cast<int>(x),
        y
      ));
    }
  }

  return points;
}

// Helper func: draw BEV vis
void drawBEVVis(
  cv::Mat& image,
  const cv::Mat& orig_frame,
  const BEVVisuals& bev_data
)
{
    // 1. Warp orig frame to BEV (640 x 640)
    if (image.size() != cv::Size(640, 640)) {
      image.create(
        640,
        640,
        orig_frame.type()
      );
    }

    cv::warpPerspective(
      orig_frame,
      image,
      bev_data.H_orig_to_bev,
      cv::Size(
        640,
        640
      )
    );

    if (!bev_data.valid) {
      cv::putText(
        image,
        "BEV Tracking: Waiting...",
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 0, 255),
        2
      );
      return;
    }

    int bev_h = 640;
    cv::Scalar color_left(255, 0, 0);     // Blue
    cv::Scalar color_right(255, 0, 200);  // Magenta
    cv::Scalar color_center(0, 255, 255); // Yellow
    int thickness = 4;

    // 2. Egoleft
    auto left_pts = genBEVPoints(
      bev_data.bev_left_coeffs,
      bev_h
    );
    if (left_pts.size() > 1) {
      cv::polylines(
        image,
        left_pts,
        false,
        color_left,
        thickness,
        cv::LINE_AA
      );
    }

    // 3. Egoright
    auto right_pts = genBEVPoints(
      bev_data.bev_right_coeffs,
      bev_h
    );
    if (right_pts.size() > 1) {
      cv::polylines(
        image,
        right_pts,
        false,
        color_right,
        thickness,
        cv::LINE_AA
      );
    }

    // 4. Drivable corridor
    auto center_pts = genBEVPoints(
      bev_data.bev_center_coeffs,
      bev_h
    );
    if (center_pts.size() > 1) {
      cv::polylines(
        image,
        center_pts,
        false,
        color_center,
        thickness,
        cv::LINE_AA
      );
    }

    // 5. Lane width bar (last known good width)
    // Width bar vis
    if (bev_data.last_valid_width_pixels > 0) {
      int y_pos = 600;    // Near bottom
      int center_x = 320; // BEV center
      int half_width = static_cast<int>(bev_data.last_valid_width_pixels / 2.0);

      cv::Point p1(
        center_x - half_width,
        y_pos
      );
      cv::Point p2(
        center_x + half_width,
        y_pos
      );

      // Main width line
      cv::line(
        image,
        p1,
        p2,
        cv::Scalar(255, 255, 255),
        2
      );

      // End markers (2 ticks both ends)
      cv::line(
        image,
        cv::Point(p1.x, y_pos-10),
        cv::Point(p1.x, y_pos+10),
        cv::Scalar(255, 255, 255),
        2
      );
      cv::line(
        image,
        cv::Point(p2.x, y_pos-10),
        cv::Point(p2.x, y_pos+10),
        cv::Scalar(255, 255, 255),
        2
      );

      // Some texts
      std::string width_txt = cv::format(
        "Lane Width: %.0f px",
        bev_data.last_valid_width_pixels
      );
      int baseline = 0;
      cv::Size sz = cv::getTextSize(
        width_txt,
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        1,
        &baseline
      );
      cv::Point text_org(
        center_x - sz.width / 2,
        y_pos - 20
      );
      cv::putText(
        image,
        width_txt,
        text_org,
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(255, 255, 255),
        1
      );
    }

    // Title
    cv::putText(
      image,
      "BEV Tracking & Recovery",
      cv::Point(20, 40),
      cv::FONT_HERSHEY_SIMPLEX,
      1.0,
      cv::Scalar(0, 255, 0),
      2
    );

}

// Helper func to draw metric curves projected to pixel space
void drawMetricVerification(
    cv::Mat& bev_image,
    const std::vector<double>& left_metric_coeffs,
    const std::vector<double>& right_metric_coeffs
)
{
    // Constants from main.cpp (MUST MATCH EXACTLY)
    const double bev_range_m = 40.0;
    const double bev_height_px = 640.0;
    const double bev_width_px = 640.0;
    const double scale = bev_range_m / bev_height_px; // 0.0625 m/px
    const double center_x = bev_width_px / 2.0;       // 320.0
    const double origin_y = bev_height_px;            // 640.0

    auto drawCurve = [&](const std::vector<double>& coeffs, cv::Scalar color) {
        if (coeffs.size() < 3) return;

        // coeffs are [c0, c1, c2] for x = c0*y^2 + c1*y + c2 (in METERS)
        double c0 = coeffs[0];
        double c1 = coeffs[1];
        double c2 = coeffs[2];

        std::vector<cv::Point> points;

        for (int y_pix = 0; y_pix < 640; ++y_pix) {
            // 1. Pixel Y -> Meter Y (Longitudinal distance)
            // origin_y is bottom (0m), decreases upwards
            double y_meter = (origin_y - y_pix) * scale;

            // 2. Evaluate Polynomial in Meters
            double x_meter = c0 * y_meter * y_meter + c1 * y_meter + c2;

            // 3. Meter X -> Pixel X (Lateral offset)
            // x_meter = (x_pix - center_x) * scale
            // x_pix = x_meter / scale + center_x
            double x_pix = (x_meter / scale) + center_x;

            if (x_pix >= 0 && x_pix < 640) {
                points.push_back(cv::Point(static_cast<int>(x_pix), y_pix));
            }
        }

        if (points.size() > 1) {
            // Draw thicker line for visibility (5px) and add outline
            cv::polylines(bev_image, points, false, cv::Scalar(255, 255, 255), 7, cv::LINE_AA); // White outline
            cv::polylines(bev_image, points, false, color, 5, cv::LINE_AA); // Colored line
        }
    };

    // Draw Left in Orange
    if (!left_metric_coeffs.empty()) {
        drawCurve(left_metric_coeffs, cv::Scalar(0, 165, 255));
        cv::putText(bev_image, "Metric L (Orange)", cv::Point(20, 580), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
    }

    // Draw Right in Red
    if (!right_metric_coeffs.empty()) {
        drawCurve(right_metric_coeffs, cv::Scalar(0, 0, 255));
        cv::putText(bev_image, "Metric R (Red)", cv::Point(20, 610), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}

cv::Mat rotateSteeringWheel(const cv::Mat& img, float steering_angle_deg)
{
  // resize overlay image
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

  cv::Point2f center(resized.cols/2.0f, resized.rows/2.0f);
  cv::Mat rot = cv::getRotationMatrix2D(center, steering_angle_deg, 1.0);

  cv::Mat rotated;
  cv::warpAffine(
      resized,
      rotated,
      rot,
      resized.size(),
      cv::INTER_LINEAR,
      cv::BORDER_CONSTANT,
      cv::Scalar(0,0,0,0)  // use 0 alpha for transparency
  );

  return rotated;
}

void visualizeWheel(const cv::Mat& img, const cv::Mat& wheelImg, const int x, const int y)
{
  if (wheelImg.empty() || img.empty()) return;

  int w = wheelImg.cols;
  int h = wheelImg.rows;

  if (x < 0 || y < 0 || x + w > img.cols || y + h > img.rows) return;

  cv::Mat roi = img(cv::Rect(x, y, w, h));

  if (wheelImg.channels() == 4) { // has alpha
    cv::Mat overlay_rgb, alpha;
    std::vector<cv::Mat> channels(4);
    cv::split(wheelImg, channels);
    alpha = channels[3]; // alpha channel
    cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, overlay_rgb);

    // Convert to float 0..1
    overlay_rgb.convertTo(overlay_rgb, CV_32FC3, 1.0 / 255.0);
    roi.convertTo(roi, CV_32FC3, 1.0 / 255.0);
    alpha.convertTo(alpha, CV_32FC1, 1.0 / 255.0);

    // Create 3-channel alpha
    cv::Mat alpha3;
    cv::cvtColor(alpha, alpha3, cv::COLOR_GRAY2BGR);

    // Blend
    cv::Mat blended = overlay_rgb.mul(alpha3) + roi.mul(cv::Scalar(1.0,1.0,1.0) - alpha3);

    // Write back to base image
    blended.convertTo(img(cv::Rect(x, y, w, h)), CV_8UC3, 255.0);
  } else {
    wheelImg.copyTo(roi); // no alpha, hard paste
  }
}

void visualizeSteering(
  cv::Mat& img,
  const float steering_angle,
  const cv::Mat& rotatedPredSteeringWheelImg,
  std::optional<float> gtSteeringAngle,
  const cv::Mat& rotatedGtSteeringWheelImg)
{
  int w = img.cols;
  int h = img.rows;

  visualizeWheel(img, rotatedPredSteeringWheelImg, 10, 10);
  if (!rotatedGtSteeringWheelImg.empty()) {
    visualizeWheel(img, rotatedGtSteeringWheelImg, w - 80, 10);
  }

  // Prediction
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << steering_angle;
  std::string steeringAngleText = "Predicted angle: " + oss.str();

  cv::putText(
    img,
    steeringAngleText,
    cv::Point(10, 100),
    cv::FONT_HERSHEY_SIMPLEX,
    0.6,
    cv::Scalar(62, 202, 130),
    2
  );

  if (gtSteeringAngle.has_value() && !std::isnan(gtSteeringAngle.value())) {
    // GT
    std::ostringstream oss1;
    oss1 << std::fixed << std::setprecision(2) << gtSteeringAngle.value();
    std::string gtAngleText = "GT angle: " + oss1.str();

    cv::putText(
      img,
      gtAngleText,
      cv::Point(w - 180, 100),
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(255, 255, 255),
      2
    );
  }

}

}  // namespace autoware_pov::vision::egolanes
