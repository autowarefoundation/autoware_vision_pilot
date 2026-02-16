/**
 * @file steering_controller.cpp
 * @brief Implementation of steering controller
 */

#include "steering_control/steering_controller.hpp"

#include <iomanip>

namespace autoware_pov::vision::steering_control {

SteeringController::SteeringController(
                                       double K_p,
                                       double K_i,
                                       double K_d,
                                       double K_S)
    : K_p(K_p), K_i(K_i), K_d(K_d), K_S(K_S)
{
    std::cout << std::fixed << std::setprecision(6); // 4 decimal places
    std::cout << "Steering controller initialized with parameters:\n"
              << "  K_p: " << K_p << "\n"
              << "  K_i: " << K_i << "\n"
              << "  K_d: " << K_d << "\n"
              << "  K_S: " << K_S << std::endl;
    prev_yaw_error = 0.0;
}

double SteeringController::computeSteering(double cte, double yaw_error, double feed_forward_steering_estimate)
{
    // Combined controller:
    // - Derivative term: K_d * (yaw_error - prev_yaw_error)
    // - Stanley controller: atan(K_i * cte)
    // - Proportional term: K_p * yaw_error
    // - Curvature feedforward: -atan(curvature) * K_S
    double steering_angle = K_d * (yaw_error - prev_yaw_error)
                          + std::atan(K_i * cte)
                          + K_p * yaw_error
                          + (feed_forward_steering_estimate) * K_S;
    prev_yaw_error = yaw_error;
    return steering_angle;
}

} // namespace autoware_pov::vision::steering_control

