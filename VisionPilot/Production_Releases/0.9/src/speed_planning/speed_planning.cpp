/**
 * @file speed_planning.cpp
 * @brief Planning speed for ACC and AEB features
 * 
 * 
 */

 #include "speed_planning/speed_planning.hpp"

namespace autoware_pov::vision::speed_planning {

SpeedPlanner::SpeedPlanner(
                                       double relative_cipo_speed,
                                       double cipo_distance,
                                       double ego_speed,
                                       double absolute_cipo_speed,
                                       bool is_cipo_present)
    : relative_cipo_speed(relative_cipo_speed), cipo_distance(cipo_distance), 
    ego_speed(ego_speed), absolute_cipo_speed(absolute_cipo_speed), is_cipo_present(is_cipo_present)
{
    std::cout << std::fixed << std::setprecision(6); // 4 decimal places
    std::cout << "Speed planner initialized with parameters:\n"
              << "  relative_cipo_speed: " << relative_cipo_speed << "\n"
              << "  cipo_distance: " << cipo_distance << "\n"
              << "  ego_speed: " << ego_speed << "\n"
              << "  absolute_cipo_speed: " << absolute_cipo_speed << std::endl;
}

// Set the speed of the ego-car
SpeedPlanner::setEgoSpeed(double ego_speed){
    SpeedPlanner::ego_speed = ego_speed;
}

// Set whether or not a CIPO is present
SpeedPlanner::setIsCIPOPresent(bool is_cipo_present){
    SpeedPlanner::is_cipo_present = is_cipo_present;
}

// Set the state of the CIPO
SpeedPlanner::setCIPOState(double relative_cipo_speed, double cipo_distance){
    SpeedPlanner::relative_cipo_speed = relative_cipo_speed;
    SpeedPlanner::cipo_distance = cipo_distance;
}

// Calculate the safe following distance based on Mobileye RSS
double SpeedPlanner::calcSafeRSSDistance(){
    double cipo_absolute_speed = SpeedPlanner::relative_cipo_speed + SpeedPlanner::ego_speed

    // Mobileye RSS formula
    double safe_distance = (SpeedPlanningConstants::response_time * SpeedPlanner::ego_speed) +
    (0.5 *SpeedPlanningConstants::a_max_accel * SpeedPlanningConstants::response_time * SpeedPlanningConstants::response_time) +
    ((SpeedPlanner::ego_speed + SpeedPlanningConstants::response_time * SpeedPlanningConstants::a_max_accel) * 
    (SpeedPlanner::ego_speed + SpeedPlanningConstants::response_time * SpeedPlanningConstants::a_max_accel))/(2*SpeedPlanningConstants::a_min_brake) -
    ((cipo_absolute_speed * cipo_absolute_speed)/(2*SpeedPlanningConstants::a_max_brake))

    // Ensuring that safe distance is at least one car length
    if (safe_distance < 5.0){
        safe_distance = 5.0;
    }

    return safe_distance

}

// Calculate the ideal driving speed based on ensuring we have a safe
// following distance
double SpeedPlanner::calcIdealDrivingSpeed();
{   
    double acceleration = 0.0;
    double set_speed = SpeedPlanner::ego_speed;
    double safe_distance = 100.0;

    // If there is a lead car
    if(SpeedPlanner::is_cipo_present){

        safe_distance = SpeedPlanner::calcSafeRSSDistance();

        // We are a safe distance away from the lead car
        if (SpeedPlanner::distance >= safe_distance * 1.1 || !SpeedPlanner::is_cipo_present){
            acceleration = 1.0;
        }

        // We are too close to the lead car
        if(SpeedPlanner::distance >= 0.5 * safe_distance && distance <= 0.9*safe_distance){
            acceleration = -1.0;
        }

        // Forward Collision Warning and Aggressive Braking
        if(SpeedPlanner::distance < 0.5 * safe_distance && distance >= 0.25*safe_distance){
            acceleration = -2.5;
        }

        // Automatic Emergency Braking
        if(SpeedPlanner::distance < 0.25 * safe_distance){
            acceleration = -5;
        }
    }
    else{
        // If there is no lead car
        acceleration = 1.0;
    }
    
    // Calculate the new set speed with a 0.5s look-ahead
    set_speed = SpeedPlanner::ego_speed + acceleration*0.5;

    // Safety check to not exceed speed limit
    if(set_speed > SpeedPlanner::speed_limit){
        set_speed = SpeedPlanner::speed_limit;
    }

    // Minimum speed of 0 - reverse is not considered
    if(set_speed < 0){
        set_speed = 0.0;
    }

    return set_speed;
}

} // namespace autoware_pov::vision::speed_planning