/**
 * @file pi_controller.cpp
 * @brief PID controller implementation for longitudinal speed control
 */

#include "longitudinal/pi_controller.hpp"
#include <iostream>
#include <iomanip>

namespace autoware_pov::vision::longitudinal {

PIController::PIController(double K_p, double K_i, double K_d)
    : K_p_(K_p), K_i_(K_i), K_d_(K_d),
      integral_error_(0.0), prev_error_(0.0)
{
    std::cout << std::fixed << std::setprecision(3)
              << "Longitudinal PID Controller initialized:\n"
              << "  K_p: " << K_p_ << "\n"
              << "  K_i: " << K_i_ << "\n"
              << "  K_d: " << K_d_ << std::endl;
}

double PIController::computeEffort(double current_speed, double target_speed)
{
    double error = target_speed - current_speed;
    
    // Accumulate integral error (with anti-windup: clamp if effort saturates)
    integral_error_ += error;
    
    // PID control law: effort = K_p*e + K_i*∫e + K_d*(de/dt)
    double effort = K_p_ * error 
                  + K_i_ * integral_error_ 
                  + K_d_ * (error - prev_error_);
    
    prev_error_ = error;
    
    return effort;
}

void PIController::reset()
{
    integral_error_ = 0.0;
    prev_error_ = 0.0;
}

} // namespace autoware_pov::vision::longitudinal
