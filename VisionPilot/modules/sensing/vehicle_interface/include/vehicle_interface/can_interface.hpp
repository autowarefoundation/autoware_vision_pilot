#ifndef VISIONPILOT_CAN_INTERFACE_HPP
#define VISIONPILOT_CAN_INTERFACE_HPP

#include <string>
#include <cstdint>
#include <vehicle_interface/vehicle_interface.hpp>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

class CanInterface : public VehicleInterface
{
public:
    explicit CanInterface(const std::string& can_device);
    ~CanInterface() override;

    CanInterface(const CanInterface&) = delete;
    CanInterface& operator=(const CanInterface&) = delete;

    double read() override;
    void write(double steering, double acceleration) override;

private:
    std::string can_device_;
    int sock_fd_ = -1;
    struct sockaddr_can addr_{};
    int if_index_ = 0;

    uint8_t steer_counter_ = 0;
    uint8_t accel_counter_ = 0;

    static constexpr canid_t CAN_ID_WHEEL_SPEEDS   = 0xAA;
    static constexpr canid_t CAN_ID_STEERING_LKA   = 0x2E4;
    static constexpr canid_t CAN_ID_ACC_CONTROL    = 0x343;
    static constexpr canid_t CAN_ID_STEER_TORQUE   = 0x260;

    static constexpr double STEER_TORQUE_MAX = 1500.0;
    static constexpr double ACCEL_MAX =  2.0;
    static constexpr double ACCEL_MIN = -3.5;

    // P-controller gain: angle(rad) → torque(Nm).  400 is a conservative
    // starting point; increase toward 800 after bench/road tuning.
    static constexpr double STEER_ANGLE_TO_TORQUE = 400.0;
};

#endif //VISIONPILOT_CAN_INTERFACE_HPP
