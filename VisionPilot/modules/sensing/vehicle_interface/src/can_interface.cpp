#include <vehicle_interface/can_interface.hpp>

#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>

#include <logging/logger.hpp>

// ── Toyota TSS2 checksum (XOR of all preceding bytes) ─────────────────────
static uint8_t toyota_checksum(const struct can_frame& frame, int len)
{
    uint8_t cs = 0;
    for (int i = 0; i < len; ++i)
        cs ^= frame.data[i];
    return cs;
}

// ── Helpers to read little-endian int16 from raw CAN bytes ────────────────
static inline int16_t read_i16(const uint8_t* d, int byte)
{
    return static_cast<int16_t>(d[byte] | (d[byte + 1] << 8));
}

static inline uint16_t read_u16(const uint8_t* d, int byte)
{
    return static_cast<uint16_t>(d[byte] | (d[byte + 1] << 8));
}

// ============================================================================
//  Construction / destruction
// ============================================================================

CanInterface::CanInterface(const std::string& can_device)
    : can_device_(can_device)
{
    sock_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock_fd_ < 0)
        throw std::runtime_error("CanInterface: socket() failed — are you root?");

    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, can_device.c_str(), IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = '\0';
    if (ioctl(sock_fd_, SIOCGIFINDEX, &ifr) < 0)
    {
        close(sock_fd_);
        sock_fd_ = -1;
        throw std::runtime_error(
            "CanInterface: interface '" + can_device + "' not found — "
            "load Red Panda Linux driver (modprobe panda) and bring can up");
    }
    if_index_ = ifr.ifr_ifindex;

    std::memset(&addr_, 0, sizeof(addr_));
    addr_.can_family = AF_CAN;
    addr_.can_ifindex = if_index_;

    if (bind(sock_fd_, reinterpret_cast<struct sockaddr*>(&addr_), sizeof(addr_)) < 0)
    {
        close(sock_fd_);
        sock_fd_ = -1;
        throw std::runtime_error("CanInterface: bind() failed on " + can_device);
    }

    // Set a short read timeout so read() doesn't block forever
    struct timeval tv{};
    tv.tv_sec  = 0;
    tv.tv_usec = 50000;  // 50 ms
    setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    VP_INFO("CanInterface: opened %s (ifindex=%d)", can_device.c_str(), if_index_);
}

CanInterface::~CanInterface()
{
    if (sock_fd_ >= 0)
        close(sock_fd_);
}

// ============================================================================
//  Read — vehicle speed from WHEEL_SPEEDS (CAN ID 0xAA)
// ============================================================================
//
//  DBC signal layout (little-endian Intel byte order):
//    WHEEL_SPEED_FR : 7|16@0+ (0.01,-67.67)   bytes 0-1
//    WHEEL_SPEED_FL : 23|16@0+ (0.01,-67.67)  bytes 2-3
//    WHEEL_SPEED_RR : 39|16@0+ (0.01,-67.67)  bytes 4-5
//    WHEEL_SPEED_RL : 55|16@0+ (0.01,-67.67)  bytes 6-7
//
//  Returns speed in m/s.  Uses 100 ms budget: keeps reading until we
//  get a WHEEL_SPEEDS frame or time out (returns last known speed).

double CanInterface::read()
{
    static double last_speed = 0.0;

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(100);

    while (std::chrono::steady_clock::now() < deadline)
    {
        struct can_frame frame{};
        const ssize_t n = ::read(sock_fd_, &frame, sizeof(frame));
        if (n < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            VP_WARN("CanInterface: read error %d", errno);
            break;
        }

        if (frame.can_id != CAN_ID_WHEEL_SPEEDS)
            continue;

        // Decode four wheel speeds (kph) and average
        const double fr = read_u16(frame.data, 0) * 0.01 - 67.67;
        const double fl = read_u16(frame.data, 2) * 0.01 - 67.67;
        const double rr = read_u16(frame.data, 4) * 0.01 - 67.67;
        const double rl = read_u16(frame.data, 6) * 0.01 - 67.67;

        const double kph = std::max({0.0, fr, fl, rr, rl});
        last_speed = kph / 3.6;   // km/h → m/s
        return last_speed;
    }

    return last_speed;
}

// ============================================================================
//  Write — steering (STEERING_LKA 0x2E4) + acceleration (ACC_CONTROL 0x343)
// ============================================================================

void CanInterface::write(double steering, double acceleration)
{
    // steering comes from the planner as a front-wheel angle (rad).
    // Convert to EPS torque command via simple P-controller.
    const double torque = std::clamp(
        steering * STEER_ANGLE_TO_TORQUE,
        -STEER_TORQUE_MAX, STEER_TORQUE_MAX);

    // ── STEERING_LKA (CAN ID 0x2E4, 5 bytes) ──────────────────────────────
    //
    //  Byte 0 : [7] SET_ME_1=1  [6:1] COUNTER  [0] STEER_REQUEST
    //  Byte 1-2 : STEER_TORQUE_CMD (int16 LE)
    //  Byte 3 : LKA_STATE = 0
    //  Byte 4 : CHECKSUM (XOR of bytes 0-3)
    {
        struct can_frame frame{};
        frame.can_id  = CAN_ID_STEERING_LKA;
        frame.can_dlc = 5;

        const int16_t torque_cmd = static_cast<int16_t>(torque);

        frame.data[0] = (1u << 7) |
                         ((steer_counter_ & 0x3F) << 1) |
                         (torque_cmd != 0 ? 1u : 0u);
        frame.data[1] = static_cast<uint8_t>(torque_cmd & 0xFF);
        frame.data[2] = static_cast<uint8_t>((torque_cmd >> 8) & 0xFF);
        frame.data[3] = 0x00;   // LKA_STATE
        frame.data[4] = toyota_checksum(frame, 4);

        if (::write(sock_fd_, &frame, sizeof(frame)) < 0)
            VP_WARN("CanInterface: STEERING_LKA write failed");

        steer_counter_ = (steer_counter_ + 1) & 0x3F;
    }

    // ── ACC_CONTROL (CAN ID 0x343, 8 bytes) ───────────────────────────────
    //
    //  Byte 0-1 : ACCEL_CMD (int16 LE, factor 0.001)
    //  Byte 2 : [7:6] ACC_TYPE=01 [5] MINI_CAR [4] DISTANCE
    //           [3] RADAR_DIRTY [2] ACC_MALFUNCTION [1:0] ALLOW_LONG_PRESS
    //  Byte 3 : [7] RELEASE_STANDSTILL [6] PERMIT_BRAKING
    //           [5] LEAD_VEHICLE_STOPPED [4] ACC_CUT_IN
    //           [3] CANCEL_REQ [2:0] unused
    //  Byte 4-5 : unused (reserved)
    //  Byte 6 : ACCEL_CMD_ALT (int8 LE, factor 0.05)
    //  Byte 7 : CHECKSUM (XOR of bytes 0-6)
    {
        struct can_frame frame{};
        frame.can_id  = CAN_ID_ACC_CONTROL;
        frame.can_dlc = 8;

        const int16_t accel_raw =
            static_cast<int16_t>(std::clamp(acceleration, ACCEL_MIN, ACCEL_MAX) / 0.001);

        frame.data[0] = static_cast<uint8_t>(accel_raw & 0xFF);
        frame.data[1] = static_cast<uint8_t>((accel_raw >> 8) & 0xFF);
        frame.data[2] = (1u << 6);   // ACC_TYPE=1, all other flags off
        frame.data[3] = (1u << 7) |  // RELEASE_STANDSTILL
                         (1u << 6);   // PERMIT_BRAKING
        frame.data[4] = 0x00;
        frame.data[5] = 0x00;
        frame.data[6] = static_cast<uint8_t>(
            static_cast<int8_t>(std::clamp(acceleration, ACCEL_MIN, ACCEL_MAX) / 0.05));
        frame.data[7] = toyota_checksum(frame, 7);

        if (::write(sock_fd_, &frame, sizeof(frame)) < 0)
            VP_WARN("CanInterface: ACC_CONTROL write failed");

        accel_counter_ = (accel_counter_ + 1) & 0x0F;
    }
}
