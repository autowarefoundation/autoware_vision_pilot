/**
 * @file speed_planning.cpp
 * @brief Planning speed for ACC and AEB features
 */

#include "speed_planning/speed_planning.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace autoware_pov::vision::speed_planning {

SpeedPlanner::SpeedPlanner(
    double relative_cipo_speed,
    double cipo_distance,
    double ego_speed,
    double absolute_cipo_speed,
    bool is_cipo_present)
    : relative_cipo_speed_(relative_cipo_speed),
      cipo_distance_(cipo_distance),
      ego_speed_(ego_speed),
      absolute_cipo_speed_(absolute_cipo_speed),
      is_cipo_present_(is_cipo_present),
      speed_limit_(SpeedPlanningConstants::speed_limit),
      is_forward_collision_warning_(false),
      is_automatic_emergency_braking_(false)
{
    std::cout << std::fixed << std::setprecision(3)
              << "SpeedPlanner initialized:\n"
              << "  ego_speed:           " << ego_speed_           << " m/s\n"
              << "  cipo_distance:       " << cipo_distance_       << " m\n"
              << "  relative_cipo_speed: " << relative_cipo_speed_ << " m/s\n"
              << "  is_cipo_present:     " << is_cipo_present_     << "\n";
}

void SpeedPlanner::setEgoSpeed(double ego_speed) {
    ego_speed_ = ego_speed;
}

void SpeedPlanner::setIsCIPOPresent(bool is_cipo_present) {
    is_cipo_present_ = is_cipo_present;
}

void SpeedPlanner::setCIPOState(double relative_cipo_speed, double cipo_distance) {
    relative_cipo_speed_ = relative_cipo_speed;
    cipo_distance_       = cipo_distance;
}

// Mobileye RSS safe distance formula:
//   d_min = v_ego * ρ + 0.5 * a_accel * ρ²
//         + (v_ego + ρ * a_accel)² / (2 * β_min)
//         − v_cipo² / (2 * β_max)
// where β_min, β_max are positive deceleration magnitudes.
double SpeedPlanner::calcSafeRSSDistance() {
    double cipo_abs_speed = ego_speed_ + relative_cipo_speed_;

    double v_after_reaction = ego_speed_ + SpeedPlanningConstants::response_time
                                         * SpeedPlanningConstants::a_max_accel;

    double safe_distance =
        (SpeedPlanningConstants::response_time * ego_speed_) +
        (0.5 * SpeedPlanningConstants::a_max_accel *
         SpeedPlanningConstants::response_time * SpeedPlanningConstants::response_time) +
        (v_after_reaction * v_after_reaction) / (2.0 * SpeedPlanningConstants::a_min_brake) -
        (cipo_abs_speed * cipo_abs_speed)     / (2.0 * SpeedPlanningConstants::a_max_brake);

    // Floor at one car length to avoid collapsing to zero at low speeds
    if (safe_distance < 5.0) {
        safe_distance = 5.0;
    }

    return safe_distance;
}

double SpeedPlanner::calcIdealDrivingSpeed() {
    double acceleration = 0.0;
    double set_speed    = ego_speed_;

    is_forward_collision_warning_  = false;
    is_automatic_emergency_braking_ = false;

    if (is_cipo_present_) {
        double d_safe = calcSafeRSSDistance();

        if (cipo_distance_ >= d_safe * 1.1) {
            // Comfortable gap — accelerate toward set speed
            acceleration = 1.0;
        } else if (cipo_distance_ >= d_safe * 0.5 && cipo_distance_ <= d_safe * 0.9) {
            // Slightly inside safe bubble — soft brake
            acceleration = -1.0;
        } else if (cipo_distance_ >= d_safe * 0.25 && cipo_distance_ < d_safe * 0.5) {
            // Forward Collision Warning zone
            acceleration = -2.5;
            is_forward_collision_warning_ = true;
        } else if (cipo_distance_ < d_safe * 0.25) {
            // Automatic Emergency Braking zone
            acceleration = -5.0;
            is_forward_collision_warning_  = true;
            is_automatic_emergency_braking_ = true;
        }
        // 0.9 ≤ d/d_safe < 1.1 → no change (hold current speed)
    } else {
        // No lead car — accelerate
        acceleration = 1.0;
    }

    // 0.5 s look-ahead integration
    set_speed = ego_speed_ + acceleration * 0.5;

    // Clamp to [0, speed_limit]
    set_speed = std::max(0.0, std::min(set_speed, speed_limit_));

    return set_speed;
}

bool SpeedPlanner::getFCWState() {
    return is_forward_collision_warning_;
}

bool SpeedPlanner::getAEBState() {
    return is_automatic_emergency_braking_;
}

} // namespace autoware_pov::vision::speed_planning
