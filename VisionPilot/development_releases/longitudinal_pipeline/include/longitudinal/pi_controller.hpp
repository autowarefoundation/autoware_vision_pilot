/**
 * @file pi_controller.hpp
 * @brief PID controller for longitudinal speed control
 * 
 * Computes acceleration/deceleration effort based on speed error.
 * Used to track ideal_speed_ms from SpeedPlanner.
 */

#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

namespace autoware_pov::vision::longitudinal {

class PIController
{
public:
    /**
     * @brief Construct PID controller with gains
     * @param K_p Proportional gain
     * @param K_i Integral gain
     * @param K_d Derivative gain
     */
    PIController(double K_p, double K_i, double K_d);

    /**
     * @brief Compute acceleration/deceleration effort (m/s²)
     * @param current_speed Current ego vehicle speed (m/s)
     * @param target_speed Desired speed from SpeedPlanner (m/s)
     * @return Control effort: positive = accelerate, negative = decelerate (m/s²)
     */
    double computeEffort(double current_speed, double target_speed);

    /**
     * @brief Reset controller state (useful after cut-in or emergency stop)
     */
    void reset();

private:
    double K_p_;
    double K_i_;
    double K_d_;
    double integral_error_;
    double prev_error_;
};

} // namespace autoware_pov::vision::longitudinal
