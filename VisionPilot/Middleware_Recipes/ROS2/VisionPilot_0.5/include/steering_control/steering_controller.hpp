/**
 * @file steering_controller.hpp
 * @brief Steering controller for path following
 * 
 * Combines Stanley controller (PI on CTE) with curvature feedforward
 * Input: CTE, yaw_error, forward_velocity, curvature (from PathFinder)
 * Output: Steering angle (radians)
 */

#pragma once

#include <iostream>
#include <cmath>

namespace autoware_pov::vision::steering_control {

/**
 * @brief Default steering controller parameters
 */

namespace SteeringControllerDefaults {    
    constexpr double K_P = 0.8;                // Proportional gain for yaw error
    constexpr double K_I = 1.6;                // Integral gain for CTE (Stanley controller)
    constexpr double K_D = 1.0;             // Derivative gain for yaw error
    constexpr double K_S = 1.0;              //proportionality constant for curvature  
};

class SteeringController
{
public:
    /**
     * @brief Constructor
     * @param K_p Proportional gain for yaw error
     * @param K_i Integral gain for CTE (Stanley controller)
     * @param K_d Derivative gain for yaw error
     * @param K_S Proportionality constant for curvature feedforward
     */
    SteeringController(
                       double K_p,
                       double K_i,
                       double K_d,
                       double K_S);
                      
    
    /**
     * @brief Compute steering angle
     * @param cte Cross-track error (meters)
     * @param yaw_error Yaw error (radians)
     * @param feed_forward_steering_estimate (angle in degrees)
     * @return Steering angle (radians)
     */
    double computeSteering(double cte, double yaw_error, double feed_forward_steering_estimate);

private:
    double K_p, K_i, K_d, K_S;
    double prev_yaw_error;
};

} // namespace autoware_pov::vision::steering_control

