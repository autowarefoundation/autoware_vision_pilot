/**
 * @file speed_planning.hpp
 * @brief Planning speed for ACC and AEB features
 * 
 */

#pragma once

namespace autoware_pov::vision::speed_planning {

/**
 * @brief Default speed planning parameters
 */

namespace SpeedPlanningConstants {    
    constexpr double a_max_brake = -4.5;             // Max deceleration of lead car 
    constexpr double a_max_accel = 2.0;              // Max acceleration of ego car
    constexpr double a_min_brake = -1.0;             // Minimum deceleration of ego car
    constexpr double response_time = 0.1;            // Reaction time of stack (10Hz)
    constexpr double speed_limit = 31.0;             // Speed limit of the road
};

class SpeedPlanner
{
public:
    /**
     * @brief Constructor
     * @param relative_cipo_speed   // Relative speed of CIPO
     * @param cipo_distance         // Distance of CIPO
     * @param ego_speed             // Absolute speed of ego car
     * @param absolute_cipo_speed   // Absolute speed of CIPO
     * @param is_cipo_present       // Bool indicating whether ther is a CIPO
     */
    SpeedPlanner(
                       double relative_cipo_speed,
                       double cipo_distance,
                       double ego_speed,
                       double absolute_cipo_speed,
                       bool is_cipo_present);
                      
    /**
     * @brief Set the speed of the ego car
     */
    setEgoSpeed(double ego_speed);

    /**
     * @brief Set whether or not there is a CIPO
     */
    setIsCIPOPresent(bool is_cipo_present);

    /**
     * @brief Set the state of the CIPO
     */
    setCIPOState(double relative_cipo_speed, double cipo_distance);

    /**
     * @brief Calculate the safe longitudinal following distance to the CIPO
     * @return Minimum safe distance of the lead car
     */
    double calcSafeRSSDistance();

    /**
     * @brief Get forward collision warning state
     * @return Check if a forward collision warning is issued
     */
    bool getFCWState();

    /**
     * @brief Get automatic emergency braking state
     * @return Check if automatic emergency braking is ocurring
     */
    bool getAEBState();


private:
    double relative_cipo_speed
    double cipo_distance;
    double ego_speed;
    double absolute_cipo_speed;
    bool is_forward_collision_warning;
    bool is_automatic_emergency_braking;
};

} // namespace autoware_pov::vision::speed_planning