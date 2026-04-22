//
// Created by atanasko on 2/12/26.
//

#pragma once

namespace visionpilot::publisher
{
struct ControlState
{
  double steering;
  double velocity;
};

class IpcSharedState
{
public:
  IpcSharedState(const char * name);
  ~IpcSharedState();

  ControlState * get();
  void set(double steering, double velocity);

private:
  const char * name_;
  int fd_;
  void * ptr_;
};
} // namespace visionpilot::publisher