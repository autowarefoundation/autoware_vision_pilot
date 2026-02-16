/**
 * @file path_finder.cpp
 * @brief PathFinder implementation
 * 
 * Adapted from PATHFINDER's cb_drivCorr() and timer_callback()
 */

#include "path_planning/path_finder.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <limits>

namespace autoware_pov::vision::path_planning {

PathFinder::PathFinder(double default_lane_width)
    : default_lane_width_(default_lane_width)
{
    initializeBayesFilter();
}

void PathFinder::initializeBayesFilter()
{
    // Configure fusion groups (from PATHFINDER pathfinder_node constructor)
    // Fusion rules: {start_idx, end_idx} → fuse indices in [start, end) → result at end_idx
    bayes_filter_.configureFusionGroups({
        {0, 3},   // CTE: fuse [0,1,2] (path,left,right) → [3] (fused)
        {5, 7},   // Yaw: fuse [5,6] (left,right) → [7] (fused)
        {9, 11}   // Curvature: fuse [9,10] (left,right) → [11] (fused)
    });
    
    // Initialize state (large variance = uncertain)
    Gaussian default_state = {0.0, 1e3};
    std::array<Gaussian, STATE_DIM> init_state;
    init_state.fill(default_state);
    
    // Initialize lane width with reasonable prior
    init_state[12].mean = default_lane_width_;
    init_state[12].variance = 0.5 * 0.5;
    
    bayes_filter_.initialize(init_state);
    
    std::cout << "[PathFinder] Initialized with default lane width: " 
              << default_lane_width_ << " m" << std::endl;
}


PathFinderOutput PathFinder::update(
    const std::vector<cv::Point2f>& left_pts_bev,
    const std::vector<cv::Point2f>& right_pts_bev,
    double autosteer_steering_rad)
{
    PathFinderOutput output;
    output.left_valid = false;
    output.right_valid = false;
    output.fused_valid = false;
    
    // 1. Predict step (add process noise, like timer_callback in PATHFINDER)
    std::array<Gaussian, STATE_DIM> process;
    std::random_device rd;
    std::default_random_engine generator(rd());
    const double epsilon = 0.00001;
    std::uniform_real_distribution<double> dist(-epsilon, epsilon);
    
    for (size_t i = 0; i < STATE_DIM; ++i) {
        process[i].mean = dist(generator);
        process[i].variance = PROC_SD * PROC_SD;
    }
    bayes_filter_.predict(process);
    
    // 2. BEV points already provided (no transformation needed)
    
    // 3. Fit polynomials to BEV points (in meters)
    auto left_coeff = fitQuadPoly(left_pts_bev);
    auto right_coeff = fitQuadPoly(right_pts_bev);
    
    FittedCurve left_curve(left_coeff);
    FittedCurve right_curve(right_coeff);
    
    output.left_coeff = left_coeff;
    output.right_coeff = right_coeff;
    output.left_valid = !std::isnan(left_curve.cte);
    output.right_valid = !std::isnan(right_curve.cte);
    
    // Store individual metrics
    output.left_cte = left_curve.cte;
    output.left_yaw_error = left_curve.yaw_error;
    output.left_curvature = left_curve.curvature;
    output.right_cte = right_curve.cte;
    output.right_yaw_error = right_curve.yaw_error;
    output.right_curvature = right_curve.curvature;
    
    // Use AutoSteer steering angle (in radians) instead of computed curvature if provided
    double steering_value =  autosteer_steering_rad ;
    
    // 4. Create measurement (adapted from cb_drivCorr)
    std::array<Gaussian, STATE_DIM> measurement;
    
    // Set measurement variances
    measurement[0].variance = STD_M_CTE * STD_M_CTE;
    measurement[1].variance = STD_M_CTE * STD_M_CTE;
    measurement[2].variance = STD_M_CTE * STD_M_CTE;
    measurement[3].variance = STD_M_CTE * STD_M_CTE;
    
    measurement[4].variance = STD_M_YAW * STD_M_YAW;
    measurement[5].variance = STD_M_YAW * STD_M_YAW;
    measurement[6].variance = STD_M_YAW * STD_M_YAW;
    measurement[7].variance = STD_M_YAW * STD_M_YAW;
    
    measurement[8].variance = STD_M_CURV * STD_M_CURV;
    measurement[9].variance = STD_M_CURV * STD_M_CURV;
    measurement[10].variance = STD_M_CURV * STD_M_CURV;
    measurement[11].variance = STD_M_CURV * STD_M_CURV;
    
    measurement[12].variance = STD_M_WIDTH * STD_M_WIDTH;
    measurement[13].variance = STD_M_WIDTH * STD_M_WIDTH;
    
    // Get current tracked width for CTE offset
    auto width = bayes_filter_.getState()[12].mean;
    
    // Set measurement means
    // [0,4,8] = ego path (we don't have it, set to NaN)
    measurement[0].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[4].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[8].mean = std::numeric_limits<double>::quiet_NaN();
    
    // [1,5,9] = left lane (offset CTE to lane center)
    measurement[1].mean = left_curve.cte + width / 2.0;
    measurement[5].mean = left_curve.yaw_error;
    measurement[9].mean =  steering_value ;
    
    // [2,6,10] = right lane (offset CTE to lane center)
    measurement[2].mean = right_curve.cte - width / 2.0;
    measurement[6].mean = right_curve.yaw_error;
    measurement[10].mean =  steering_value ;
    
    // [3,7,11] = fused (computed by Bayes filter)
    measurement[3].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[7].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[11].mean = std::numeric_limits<double>::quiet_NaN();
    
    // Lane width measurement (adapted from cb_drivCorr logic)
    if (std::isnan(left_curve.cte) && std::isnan(right_curve.cte)) {
        // Both lanes missing → use default
        measurement[12].mean = default_lane_width_;
    } else if (std::isnan(left_curve.cte)) {
        // Left missing → keep current tracked width
        measurement[12].mean = width;
    } else if (std::isnan(right_curve.cte)) {
        // Right missing → keep current tracked width
        measurement[12].mean = width;
    } else {
        // Both present → direct measurement
        measurement[12].mean = right_curve.cte - left_curve.cte;
    }
    
    measurement[13].mean = std::numeric_limits<double>::quiet_NaN();
    
    // 5. Update Bayes filter
    bayes_filter_.update(measurement);
    
    // 6. Extract fused state
    const auto& state = bayes_filter_.getState();
    
    output.cte = state[3].mean;
    output.yaw_error = state[7].mean;
    // Use AutoSteer steering angle (radians) if provided, otherwise use fused curvature from Bayes filter
    output.curvature =  autosteer_steering_rad ;
    output.lane_width = state[12].mean;
    
    output.cte_variance = state[3].variance;
    output.yaw_variance = state[7].variance;
    output.curv_variance = state[11].variance;
    output.lane_width_variance = state[12].variance;
    
    output.fused_valid = !std::isnan(output.cte) && 
                         !std::isnan(output.yaw_error) && 
                         !std::isnan(output.curvature);
    
    return output;
}

const std::array<Gaussian, STATE_DIM>& PathFinder::getState() const
{
    return bayes_filter_.getState();
}

void PathFinder::reset()
{
    initializeBayesFilter();
}

} // namespace autoware_pov::vision::path_planning

