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
     */
    SpeedPlanner(
                       double relative_cipo_speed,
                       double cipo_distance,
                       double ego_speed,
                       double absolute_cipo_speed);
                      
    /**
     * @brief Set the speed of the ego car
     */
    setEgoSpeed(double ego_speed);

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
     * @brief Calculate the driving speed the ego car should follow
     * @return Driving speed which is sent to the speed controller to track
     */
    double calcIdealDrivingSpeed();

private:
    double relative_cipo_speed
    double cipo_distance;
    double ego_speed;
    double absolute_cipo_speed;
};

} // namespace autoware_pov::vision::speed_planning