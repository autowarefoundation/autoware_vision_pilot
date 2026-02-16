//
// Created by atanasko on 12/14/25.
//

#include "steering_control/steering_filter.hpp"
#include <numeric>
#include <opencv2/core.hpp>

namespace autoware_pov::vision::steering_control
{
SteeringFilter::SteeringFilter(const float smoothing_factor, float initial)
  : tau_(smoothing_factor), previous_steering(initial)
{
  steering_angle_buffer.set_capacity(10);
}

float SteeringFilter::filter(const float current_steering, const float dt)
{
  steering_angle_buffer.push_back(current_steering);
  // const float alpha = dt / (tau_ + dt);
  // const float alpha = 0.9;

  // float clamp_steering = 0.0;
  // float delta_steering = current_steering - previous_steering;
  // if (delta_steering > 0.2) {
  //   clamp_steering = previous_steering + 0.75;
  // }
  //
  // if (delta_steering < -0.2) {
  //   clamp_steering = previous_steering - 0.75;
  // }
  // previous_steering = alpha * current_steering + (1.0f - alpha) * previous_steering;

  float running_sum = std::accumulate(
    steering_angle_buffer.begin(), steering_angle_buffer.end(), 0LL);
  float filtered_steering_value = running_sum / steering_angle_buffer.size();

  return filtered_steering_value;
}

void SteeringFilter::reset(float value)
{
  previous_steering = value;
}
} // namespace autoware_pov::vision::steering_control