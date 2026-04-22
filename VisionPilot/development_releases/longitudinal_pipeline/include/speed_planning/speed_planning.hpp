/**
 * @file speed_planning.hpp
 * @brief Planning speed for ACC and AEB features
 * 
 */

#pragma once

namespace autoware_pov::vision::speed_planning {

/**
 * @brief Speed planning constants derived from Mobileye RSS model.
 * 
 * Brake constants are positive magnitudes (m/s²).
 * The RSS safe distance formula subtracts CIPO stopping contribution from ego stopping requirement.
 */
namespace SpeedPlanningConstants {
    constexpr double a_max_brake   = 4.5;   // Max deceleration capability of lead car (CIPO), m/s²
    constexpr double a_max_accel   = 2.0;   // Max acceleration of ego car, m/s²
    constexpr double a_min_brake   = 1.0;   // Min deceleration of ego car (comfortable), m/s²
    constexpr double response_time = 0.1;   // System response time (10 Hz pipeline), s
    constexpr double speed_limit   = 31.0;  // Road speed limit, m/s (~70 mph)
}

class SpeedPlanner
{
public:
    /**
     * @brief Construct SpeedPlanner with initial state.
     * @param relative_cipo_speed  Relative speed of CIPO wrt ego (from Kalman), m/s
     * @param cipo_distance        Distance to CIPO, m
     * @param ego_speed            Absolute ego-car speed (from CAN), m/s
     * @param absolute_cipo_speed  Absolute CIPO speed (ego + relative), m/s
     * @param is_cipo_present      Whether a valid CIPO is tracked
     */
    SpeedPlanner(
        double relative_cipo_speed,
        double cipo_distance,
        double ego_speed,
        double absolute_cipo_speed,
        bool is_cipo_present);

    /** @brief Update ego-car speed each frame (from CAN bus). */
    void setEgoSpeed(double ego_speed);

    /** @brief Update whether a CIPO is currently present. */
    void setIsCIPOPresent(bool is_cipo_present);

    /**
     * @brief Update CIPO state each frame (from ObjectFinder / Kalman).
     * @param relative_cipo_speed  Relative velocity: positive = closing, negative = opening
     * @param cipo_distance        Measured distance to CIPO, m
     */
    void setCIPOState(double relative_cipo_speed, double cipo_distance);

    /**
     * @brief Compute RSS safe following distance based on current state.
     * @return Minimum safe distance d_min (m), floored at one car length (5 m)
     */
    double calcSafeRSSDistance();

    /**
     * @brief Compute ideal driving speed using tiered acceleration policy.
     *
     * Policy relative to d / d_min ratio:
     *   ≥ 1.1   → accelerate  (+a_max_accel)
     *   0.5–0.9 → soft brake  (-1.0 m/s²)
     *   0.25–0.5 → FCW + hard brake (-2.5 m/s²)
     *   < 0.25  → AEB         (-5.0 m/s²)
     *
     * @return Commanded set-speed (m/s), clamped to [0, speed_limit]
     */
    double calcIdealDrivingSpeed();

    /** @brief Returns true if a Forward Collision Warning is currently active. */
    bool getFCWState();

    /** @brief Returns true if Automatic Emergency Braking is currently commanded. */
    bool getAEBState();

private:
    double relative_cipo_speed_;
    double cipo_distance_;
    double ego_speed_;
    double absolute_cipo_speed_;
    bool   is_cipo_present_;
    double speed_limit_;
    bool   is_forward_collision_warning_;
    bool   is_automatic_emergency_braking_;
};

} // namespace autoware_pov::vision::speed_planning
