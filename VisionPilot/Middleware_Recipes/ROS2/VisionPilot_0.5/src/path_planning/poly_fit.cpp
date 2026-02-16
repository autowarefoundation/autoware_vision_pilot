/**
 * @file poly_fit.cpp
 * @brief Polynomial fitting implementation
 * 
 * Extracted from PATHFINDER package's path_finder.cpp
 */

#include "path_planning/poly_fit.hpp"
#include <Eigen/Dense>
#include <limits>
#include <cmath>

namespace autoware_pov::vision::path_planning {

FittedCurve::FittedCurve() 
    : coeff{std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()},
      cte(std::numeric_limits<double>::quiet_NaN()),
      yaw_error(std::numeric_limits<double>::quiet_NaN()),
      curvature(std::numeric_limits<double>::quiet_NaN())
{
}

FittedCurve::FittedCurve(const std::array<double, 3> &coeff) : coeff(coeff)
{
    // Derived metrics at y=0 (vehicle position)
    cte = -coeff[2];  // Lateral offset
    yaw_error = -std::atan2(coeff[1], 1.0);  // Heading angle
    
    // Curvature is no longer computed here - we use AutoSteer steering angle instead
    // Keep curvature field for backward compatibility with Bayes filter (will use AutoSteer when available)
    curvature = std::numeric_limits<double>::quiet_NaN();
}

std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points)
{
    const int degree = 2;
    const size_t N = points.size();
    
    // Need at least 3 points for quadratic fit
    if (N <= 2)
    {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }

    // Setup least squares: A * [c0, c1, c2]^T = b
    // where x = c0*y² + c1*y + c2
    Eigen::MatrixXd A(N, degree + 1);
    Eigen::VectorXd b(N);

    for (size_t i = 0; i < N; ++i)
    {
        double x = points[i].x;  // Lateral position
        double y = points[i].y;  // Longitudinal position

        A(i, 0) = y * y;  // y² term
        A(i, 1) = y;      // y term
        A(i, 2) = 1.0;    // constant term
        b(i) = x;         // Target
    }
    
    // Solve using QR decomposition
    Eigen::VectorXd res = A.colPivHouseholderQr().solve(b);
    
    std::array<double, 3> coeffs;
    for (int i = 0; i <= degree; ++i)
    {
        coeffs[i] = res(i);
    }
    
    return coeffs;
}

} // namespace autoware_pov::vision::path_planning

