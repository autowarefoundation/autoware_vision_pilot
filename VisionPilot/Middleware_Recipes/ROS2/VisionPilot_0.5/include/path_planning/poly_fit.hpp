/**
 * @file poly_fit.hpp
 * @brief Polynomial fitting and lane curve utilities
 * 
 * Extracted from PATHFINDER package's path_finder.hpp/cpp
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

namespace autoware_pov::vision::path_planning {

/**
 * @brief Fitted quadratic curve: x = c0*y² + c1*y + c2
 * 
 * Vehicle-centric BEV coordinates:
 * - x: lateral position (positive = right)
 * - y: longitudinal position (positive = forward)
 */
struct FittedCurve
{
    std::array<double, 3> coeff; // [c0, c1, c2] for x = c0*y² + c1*y + c2
    double cte;                  // Cross-track error at y=0 (meters)
    double yaw_error;            // Yaw error at y=0 (radians)
    double curvature;            // Curvature at y=0 (meters^-1)
    
    FittedCurve();
    explicit FittedCurve(const std::array<double, 3> &coeff);
};

/**
 * @brief Fit quadratic polynomial to BEV points
 * 
 * Fits x = c0*y² + c1*y + c2 using least squares
 * 
 * @param points BEV points in meters (x=lateral, y=longitudinal)
 * @return Coefficients [c0, c1, c2], or NaN if insufficient points
 */
std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points);

} // namespace autoware_pov::vision::path_planning

