#ifndef AUTOWARE_POV_DRIVERS_CAN_INTERFACE_HPP_
#define AUTOWARE_POV_DRIVERS_CAN_INTERFACE_HPP_

#include <string>
#include <optional>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <memory>

namespace autoware_pov::drivers {

struct CanVehicleState {
    double speed_kmph = std::numeric_limits<double>::quiet_NaN();           // Speed, CAN ID 0xA1
    double steering_angle_deg = std::numeric_limits<double>::quiet_NaN();   // Steering, CAN ID 0xA4
    bool is_valid = false;
    bool is_steering_angle = false;

    void clear()
    {
      speed_kmph = std::numeric_limits<double>::quiet_NaN();
      is_steering_angle = false;
      steering_angle_deg = std::numeric_limits<double>::quiet_NaN();
    }
};

/**
 * @brief CAN Interface for hardware (SocketCAN) and file replay (.asc)
 * - If interface_name ends with ".asc" : file replay.
 * - Otherwise : open a SocketCAN interface (real-time inference).
 */
class CanInterface {

public:

    /**
     * @brief Constructor
     * @param interface_name "can0", "vcan0", or path to .asc file (e.g. "./assets/test.asc")
     */
    explicit CanInterface(const std::string& interface_name);

    ~CanInterface();

    /**
     * @brief Read available CAN frames and update state
     * * Call this once per cycle in main loop.
     * In real-time inference : reads all frames currently in the socket buffer.
     * In fie replay : reads the next line from the ASC file.
     * * @return true if new data was processed, false otherwise
     */
    bool update();

    /**
     * @brief Get latest decoded vehicle state
     */
    CanVehicleState getState() const;

    /**
     * @brief Check if in file replay
     */
    bool isReplayMode() const { return is_file_mode_; }

private:

    // State
    CanVehicleState current_state_;
    bool is_file_mode_ = false;

    // Real-time inference (SocketCAN)
    int socket_fd_ = -1;
    void setupSocket(const std::string& iface_name);
    bool readSocket();

    // File replay (.asc)
    std::ifstream file_stream_;
    void setupFile(const std::string& file_path);
    bool readFileLine();

    // Decoding IDs
    static constexpr int ID_SPEED = 0xA1;    // Or A1, 161
    static constexpr int ID_STEERING = 0xA4; // Or A4, 164

    void parseFrame(
        int can_id,
        const std::vector<uint8_t>& data
    );

    double decodeSpeed(const std::vector<uint8_t>& data);

    double decodeSteering(const std::vector<uint8_t>& data);
    
};

// extern declaration
extern std::unique_ptr<CanInterface> can_interface;

} // namespace autoware_pov::drivers

#endif // AUTOWARE_POV_DRIVERS_CAN_INTERFACE_HPP_