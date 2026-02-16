// This script takes the output of the lane fitler.

// The blue and pink perspective image polynomial fitted lines need to be sampled and the samples should be transformed into a BEV space using a Homography transform (please note - we are not creating a BEV image, only transforming discrete points from the perspective coordinates to the BEV coordinates).

// To do this, you can simply multiply the coordinates of the polyfit line samples by the Homography matrix.

// Verify this is working by visualizing the transformed points.

// Once this has been verfied, we will proceed with the drivable corridor parameter estimation and temporal tracking.

#include "lane_tracking/lane_tracking.hpp"
#include <cmath>
#include <algorithm>


namespace autoware_pov::vision::egolanes
{

LaneTracker::LaneTracker() {
    // Homography is pre-computed elsewhere and hard-coded.
    // If you got any problem with it, ask me.
}

void LaneTracker::initHomography(const cv::Size& image_size) {

    if (
        homography_inited && 
        cached_image_size == image_size
    ) return;

    cached_image_size = image_size;
    homography_inited = true;
    
}

std::pair<LaneSegmentation, DualViewMetrics> LaneTracker::update(
    const LaneSegmentation& input_lanes,
    const cv::Size& image_size
) {
    
    // Ensure homography ready
    initHomography(image_size);

    LaneSegmentation output_lanes = input_lanes;
    DualViewMetrics metrics;

    metrics.bev_visuals.H_orig_to_bev = H_orig_to_bev.clone();

    // Coeffs in input_lanes are normalized to 160 x 80.
    // Must upscale em to image_size for warping.
    
    double scale_x = static_cast<double>(image_size.width) / input_lanes.width;
    double scale_y = static_cast<double>(image_size.height) / input_lanes.height;

    // Helper lambda to upscale coeffs
    auto upscaleCoeffs = [&](const std::vector<double>& c) {
        std::vector<double> up(6);
        
        // y_img = y_model * scale_y
        // x_img = x_model * scale_x
        // x_model = ay^2 + by + c
        // x_img/sx = a(y_img/sy)^2 + b(y_img/sy) + c
        // x_img = (a*sx/sy^2)*y_img^2 + (b*sx/sy)*y_img + (c*sx)

        if (c.size() < 6) return up;
        up[0] = 0; // Cubic term ignored for now if we use quadratic
        if (c.size() == 6) { // Assuming quadratic storage [0, a, b, c, min, max]
             up[1] = c[1] * scale_x / (scale_y * scale_y);
             up[2] = c[2] * scale_x / scale_y;
             up[3] = c[3] * scale_x;
             up[4] = c[4] * scale_y;
             up[5] = c[5] * scale_y;
        }
        return up;
    };

    bool left_valid = !input_lanes.left_coeffs.empty();
    bool right_valid = !input_lanes.right_coeffs.empty();

    std::vector<cv::Point2f> left_pts_bev, right_pts_bev;


    // 1. WARP EXISTING LINES TO BEV SPACE
    if (left_valid) {
        auto up_coeffs = upscaleCoeffs(input_lanes.left_coeffs);
        auto pts_pers = genPointsFromCoeffs(
            up_coeffs, 
            image_size.height
        );
        left_pts_bev = warpPoints(
            pts_pers, 
            H_orig_to_bev
        );
    }

    if (right_valid) {
        auto up_coeffs = upscaleCoeffs(input_lanes.right_coeffs);
        auto pts_pers = genPointsFromCoeffs(
            up_coeffs, 
            image_size.height
        );
        right_pts_bev = warpPoints(
            pts_pers, 
            H_orig_to_bev
        );
    }

    // 2. UPDATE LANE WIDTH HISTORY OR RECOVER MISSING LINES
    
    // a. Both lines present
    if (
        left_valid && 
        right_valid
    ) {
        
        // Update: calc average lateral distance in BEV at bottom - BEV lane width
        if (
            !left_pts_bev.empty() && 
            !right_pts_bev.empty()
        ) {
            // Simple X diff at bottom
            double w = std::abs(
                right_pts_bev.back().x - 
                left_pts_bev.back().x
            );
            
            // Smooth update so the width won't jump too much (hopefully lol)
            last_valid_bev_width = 
                (has_valid_width_history) ? 
                (last_valid_bev_width * 0.9 + w * 0.1) : 
                w;
            has_valid_width_history = true;
        }
    }

    // b. LEFT missing, RIGHT present
    else if (
        !left_valid && 
        right_valid && 
        has_valid_width_history
    ) {
        
        // Recover LEFT from RIGHT + last known width
        left_pts_bev = right_pts_bev;
        for (auto& p : left_pts_bev) {
            p.x -= last_valid_bev_width; // Shift LEFT in BEV
        }

        // Reproject back to orig to update vis lines
        auto recovered_orig = warpPoints(
            left_pts_bev, 
            H_bev_to_orig
        );
        
        // Re-fit poly in model space 160 x 80 for consistency
        // Downscale first
        std::vector<cv::Point2f> model_pts;
        for(const auto& p : recovered_orig) {
            model_pts.push_back(cv::Point2f(
                p.x / scale_x, 
                p.y / scale_y
            ));
        }
        output_lanes.left_coeffs = fitPoly2ndOrder(
            model_pts, 
            input_lanes.height
        );
    }

    // c. RIGHT missing, LEFT present
    else if (
        left_valid && 
        !right_valid && 
        has_valid_width_history
    ) {
        
        // Recover RIGHT from LEFT + last known width
        right_pts_bev = left_pts_bev;
        for (auto& p : right_pts_bev) {
            p.x += last_valid_bev_width; // Shift RIGHT in BEV
        }

        // Reproject back to orig to update vis lines
        auto recovered_orig = warpPoints(
            right_pts_bev, 
            H_bev_to_orig
        );
        
        // Re-fit poly in model space 160 x 80 for consistency
        // Downscale first
        std::vector<cv::Point2f> model_pts;
        for(const auto& p : recovered_orig) {
            model_pts.push_back(cv::Point2f(
                p.x / scale_x, 
                p.y / scale_y
            ));
        }
        output_lanes.right_coeffs = fitPoly2ndOrder(
            model_pts, 
            input_lanes.height
        );
    }

    // 3. COMPUTE CURVE PARAMS BOTH VIEWS (if at least one lane present)
    if (
        !left_pts_bev.empty() && 
        !right_pts_bev.empty()
    ) {
        
        // a. BEV DRIVING CORRIDOR

        std::vector<cv::Point2f> center_pts_bev;
        size_t n = std::min(
            left_pts_bev.size(), 
            right_pts_bev.size()
        );
        for (size_t i = 0; i < n; ++i) {
            cv::Point2f mid = (left_pts_bev[i] + right_pts_bev[i]) * 0.5f;
            center_pts_bev.push_back(mid);
        }

        // Fit curve in BEV
        auto bev_coeffs = fitPoly2ndOrder(
            center_pts_bev, 
            640
        );

        // Store coeffs for BEV vis later
        metrics.bev_visuals.bev_center_coeffs = bev_coeffs;
        metrics.bev_visuals.bev_left_coeffs = fitPoly2ndOrder(
            left_pts_bev, 
            640
        );
        metrics.bev_visuals.bev_right_coeffs = fitPoly2ndOrder(
            right_pts_bev, 
            640
        );

        // BEV curve params at bottom of BEV grid (y = 640)
        // BEV center is x = 320
        double bev_car_y = 640.0;
        metrics.bev_lane_offset = calcLaneOffset(
            bev_coeffs, 
            bev_car_y
        ) - 320.0; 
        metrics.bev_yaw_offset = calcYawOffset(
            bev_coeffs, 
            bev_car_y
        );
        metrics.bev_curvature = calcCurvature(
            bev_coeffs, 
            bev_car_y
        );

        // b. PERSPECTIVE DRIVING CORRIDOR (for vis)

        output_lanes.center_coeffs.resize(6);
        for (int i = 0; i < 6; ++i) {
             output_lanes.center_coeffs[i] = (
                output_lanes.left_coeffs[i] + 
                output_lanes.right_coeffs[i]
            ) / 2.0;
        }
        output_lanes.path_valid = true;

        // Orig curve params at bottom of orig grid)
        double orig_car_y = 79.0;
        metrics.orig_lane_offset = calcLaneOffset(
            output_lanes.center_coeffs, 
            orig_car_y
        ) - (input_lanes.width / 2.0);
        metrics.orig_yaw_offset = calcYawOffset(
            output_lanes.center_coeffs, 
            orig_car_y
        );
        metrics.orig_curvature = calcCurvature(
            output_lanes.center_coeffs, 
            orig_car_y
        );

        // Output struct curve params
        output_lanes.lane_offset = metrics.orig_lane_offset;
        output_lanes.yaw_offset = metrics.orig_yaw_offset;
        output_lanes.curvature = metrics.orig_curvature;

        // Populate BEV vis data
        metrics.bev_visuals.bev_left_pts = left_pts_bev;   // Store BEV points for PathFinder
        metrics.bev_visuals.bev_right_pts = right_pts_bev; // Store BEV points for PathFinder
        metrics.bev_visuals.last_valid_width_pixels = last_valid_bev_width;
        metrics.bev_visuals.valid = true;
    }

    return {
        output_lanes, 
        metrics
    };
}


// HELPER FUNCS

std::vector<cv::Point2f> LaneTracker::warpPoints(
    const std::vector<cv::Point2f>& src_pts, 
    const cv::Mat& H
) {

    std::vector<cv::Point2f> dst_pts;
    if (src_pts.empty()) return dst_pts;
    cv::perspectiveTransform(
        src_pts, 
        dst_pts, 
        H
    );
    return dst_pts;

}

std::vector<cv::Point2f> LaneTracker::genPointsFromCoeffs(
    const std::vector<double>& c, 
    int height,
    int step
) {

    std::vector<cv::Point2f> pts;
    if (c.size() < 6) return pts;
    
    // c is [0, a, b, c, min_y, max_y] for x = ay^2 + by + c
    // Or [a, b, c, d, min, max] for cubic
    // Now using quadratic (TODO: add dynamic logic later - Tran)
    
    double min_y = c[4];
    double max_y = c[5];

    for (double y = min_y; y <= max_y; y += step) {
        double x = 0;
        if (c[1] != 0) {    // Quadratic
            x = c[1]*y*y + c[2]*y + c[3];
        } else {            // Linear
             x = c[2]*y + c[3];
        }
        pts.push_back(cv::Point2f(
            (float)x, 
            (float)y
        ));
    }
    return pts;

}

std::vector<double> LaneTracker::fitPoly2ndOrder(
    const std::vector<cv::Point2f>& points,
    int img_height
) {

    std::vector<double> coeffs(
        6, 
        0.0
    );
    if (points.size() < 3) return coeffs;

    // Least squares for x = ay^2 + by + c
    cv::Mat A(
        points.size(), 
        3, 
        CV_64F
    );
    cv::Mat B(
        points.size(), 
        1, 
        CV_64F
    );

    double min_y = 1e9, max_y = -1e9;

    for (size_t i = 0; i < points.size(); ++i) {
        double y = points[i].y;
        double x = points[i].x;
        
        A.at<double>(i, 0) = y * y;
        A.at<double>(i, 1) = y;
        A.at<double>(i, 2) = 1.0;
        B.at<double>(i, 0) = x;

        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    cv::Mat solution;
    if (cv::solve(
        A, 
        B, 
        solution, 
        cv::DECOMP_SVD
    )) {
        coeffs[0] = 0.0;                        // Cubic term
        coeffs[1] = solution.at<double>(0); // a
        coeffs[2] = solution.at<double>(1); // b
        coeffs[3] = solution.at<double>(2); // c
        coeffs[4] = min_y;
        coeffs[5] = max_y;
    }
    return coeffs;

}

double LaneTracker::calcLaneOffset(
    const std::vector<double>& c, 
    double y
) {

    if (c.size() < 4) return 0.0;
    
    // Quadratic: ay^2 + by + c (stored as 0, a, b, c...)
    return c[1]*y*y + c[2]*y + c[3];

}

double LaneTracker::calcYawOffset(
    const std::vector<double>& c, 
    double y
) {

    if (c.size() < 4) return 0.0;
    
    // dx/dy = 2ay + b
    double dx_dy = 2*c[1]*y + c[2];

    return std::atan(dx_dy);

}

double LaneTracker::calcCurvature(
    const std::vector<double>& c, 
    double y
) {

    if (c.size() < 4) return 0.0;
    
    // x' = 2ay + b
    // x'' = 2a
    double dx_dy = 2*c[1]*y + c[2];
    double d2x_dy2 = 2*c[1];
    
    double denom = std::pow(
        1 + dx_dy*dx_dy, 
        1.5
    );
    if (std::abs(denom) < 1e-6) return 0.0;

    return std::abs(d2x_dy2) / denom;
}

} // namespace autoware_pov::vision::egolanes