//
// Created by atanasko on 12/14/25.
//

#ifndef EGOLANES_PIPELINE_LOW_PASS_FILTER_HPP
#define EGOLANES_PIPELINE_LOW_PASS_FILTER_HPP
#include <boost/circular_buffer.hpp>

namespace autoware_pov::vision::steering_control
{
class SteeringFilter
{
public:
  explicit SteeringFilter(float smoothing_factor, float initial = 0.0f);

  float filter(float current_steering, float dt);

  void reset(float value = 0.0f);

private:
  float tau_;
  float previous_steering;
  boost::circular_buffer<float> steering_angle_buffer;
};
}


#endif //EGOLANES_PIPELINE_LOW_PASS_FILTER_HPP