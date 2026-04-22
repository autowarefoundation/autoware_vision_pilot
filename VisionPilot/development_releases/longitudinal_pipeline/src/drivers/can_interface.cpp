#include "drivers/can_interface.hpp"

#include <iostream>
#include <cstring>
#include <sstream>
#include <iomanip>

// Linux SocketCAN headers
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <fcntl.h>


namespace autoware_pov::drivers {

// Constructor
CanInterface::CanInterface(
    const std::string& interface_name
) {

    // If file path (.asc) : file replay
    if (interface_name.find(".asc") != std::string::npos) {
        std::cout << "[CanInterface] Detected .asc file extension. Initializing file replay mode: " 
                  << interface_name << std::endl;
        setupFile(interface_name);
    } 
    // Else : real-time inference via SocketCAN
    else {
        std::cout << "[CanInterface] Initializing real-time inference mode (SocketCAN): " 
                  << interface_name << std::endl;
        setupSocket(interface_name);
    }
}

// Destructor
CanInterface::~CanInterface() {

    if (
        !is_file_mode_ && 
        socket_fd_ >= 0
    ) {
        close(socket_fd_);
    }

    if (
        is_file_mode_ && 
        file_stream_.is_open()
    ) {
        file_stream_.close();
    }
}

// Main update loop
bool CanInterface::update() {
    // clear because data get from state!
    current_state_.clear();

    if (is_file_mode_) {
        return readFileLine();
    } else {
        return readSocket();
    }
}

// Get latest vehicle state
CanVehicleState CanInterface::getState() const {

    return current_state_;
}

// Decoding from CAN frame
void CanInterface::parseFrame(
    int can_id, 
    const std::vector<uint8_t>& data
) {

    if (data.empty()) return;

    // ID 0xA1 (A1, 161) => Speed
    if (can_id == ID_SPEED) {
        double val = decodeSpeed(data);
        current_state_.speed_kmph = val;
        current_state_.is_valid = true;
    } 
    // ID 0xA4 (A4, 164) => Steering angle
    else if (can_id == ID_STEERING) {
        double val = decodeSteering(data);
        current_state_.steering_angle_deg = val;
        current_state_.is_valid = true;
        current_state_.is_steering_angle = true;
    }
}

// ABSSP1 (Speed) : Start Bit 39 | Length 16 | Signed | Factor 0.01
// Format: Motorola (Big Endian: https://en.wikipedia.org/wiki/Endianness)
// Bit 39 is in byte 4. (0-indexed). 
// 16 bits -> spans byte 4 (high) and byte 5 (low) or 4 and 3?
// We assume [byte 4] is MSB, [byte 5] is LSB based on standard layout.
double CanInterface::decodeSpeed(const std::vector<uint8_t>& data) {

    if (data.size() < 8) return 0.0;

    // Combine byte 4 and byte 5
    // Note: DBC bit numbering can be tricky. If start is 39 (0x27), that is bit 7 of byte 4.
    // We assume standard Big Endian placement.
    int16_t raw = (static_cast<int16_t>(data[4]) << 8) | static_cast<int16_t>(data[5]);
    
    return static_cast<double>(raw) * 0.01;
}

// Steering angle
// SSA (Measured steering angle)    : 46|15@0- (0.1,0) [0|0] "-"
// SSAZ (Steering zero point)       : 29|15@0- (0.1,0) [0|0] "-"
double CanInterface::decodeSteering(const std::vector<uint8_t>& data) {

    // Sanity: need at least 8 bytes
    if (data.size() < 8) return std::numeric_limits<double>::quiet_NaN();

    // Workflow, as desribed in DBC:
    // 1. Get steering zero point value by reading SSAZ
    // 2. Get measured steering angle by reading SSA
    // 3. Final steering angle = steering angle - steering zero point

    // 1. SSAZ
    // Byte 3: 29, 28, 27, 26, 25, 24
    // Byte 4: 39, 38, 37, 36, 35, 34, 33, 32
    // Byte 5: 47
    
    uint16_t ssaz_byte_3 = data[3] & 0x3F;                  // Bits 0-5 of byte 3 (00xx xxxx)
    uint16_t ssaz_byte_4 = data[4];                         // Byte 4
    uint16_t ssaz_byte_5 = (data[5] >> 7) & 0x01;           // Top bit of byte 5
    
    uint32_t ssaz_raw = (ssaz_byte_3 << 9) |
                        (ssaz_byte_4 << 1) |
                         ssaz_byte_5;

    // Sign extension (15-bit => 16-bit signed)
    // This is to ensure negative values are correctly interpreted
    int16_t ssaz_signed = static_cast<int16_t>(ssaz_raw << 1) >> 1;
    
    double deg_ssaz = static_cast<double>(ssaz_signed) * 0.1;  // Deg conversion

    // 2. SSA
    // Byte 5: 46, 45, 44, 43, 42, 41, 40
    // Byte 6: 55, 54, 53, 52, 51, 50, 49, 48
    
    uint16_t ssa_byte_5 = data[5] & 0x7F;                   // Bits 0-6 of byte 5 (0xxx xxxx)
    uint16_t ssa_byte_6 = data[6];                          // Byte 6
    
    uint16_t ssa_raw =  (ssa_byte_5 << 8) | 
                         ssa_byte_6;
    
    // Sign extension (15-bit => 16-bit signed)
    // This is to ensure negative values are correctly interpreted
    int16_t ssa_signed = static_cast<int16_t>(ssa_raw << 1) >> 1;
    
    double deg_ssa = static_cast<double>(ssa_signed) * 0.1;    // Deg conversion

    // 3. Final steering angle
    double steering_angle = deg_ssa - deg_ssaz;

    return steering_angle;

}

// ============================== REAL-TIME INFERENCE (SocketCAN) ============================== //

// SocketCAN setup during real-time inference
void CanInterface::setupSocket(
    const std::string& iface_name
) {

    is_file_mode_ = false;

    // 1. Open socket
    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
        perror("[CanInterface] Error opening socket");
        return;
    }

    // 2. Resolve interface index
    struct ifreq ifr;
    std::strncpy(
        ifr.ifr_name, 
        iface_name.c_str(), 
        IFNAMSIZ - 1
    );
    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
        perror("[CanInterface] Error finding interface index");
        close(socket_fd_); // Close socket on failure
        socket_fd_ = -1;   // Mark as invalid
        return;
    }

    // 3. Bind
    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[CanInterface] Error binding socket");
        close(socket_fd_); // Close socket on failure
        socket_fd_ = -1;   // Mark as invalid
        return;
    }

    // 4. Set non-blocking
    // This ensures update() doesn't hang the whole Autoware pipeline if no data comes
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
}

// Read socket
bool CanInterface::readSocket() {
    
    if (socket_fd_ < 0) return false;

    struct can_frame frame;
    bool data_received = false;

    // Read all pending frames in the buffer
    while (true) {
        int nbytes = read(
            socket_fd_, 
            &frame, 
            sizeof(struct can_frame)
        );
        if (nbytes < 0) {
            // No more data (EAGAIN) or error
            break;
        }
        if (nbytes < (int)sizeof(struct can_frame)) {
            continue; // Incomplete frame
        }

        // Vector conversion for safe handling
        std::vector<uint8_t> data_vec(
            frame.data, 
            frame.data + frame.can_dlc
        );
        parseFrame(
            frame.can_id, 
            data_vec
        );
        data_received = true;
    }
    return data_received;
}

// ============================== FILE REPLAY (.asc) ============================== //

void CanInterface::setupFile(
    const std::string& file_path
) {

    is_file_mode_ = true;
    file_stream_.open(file_path);
    if (!file_stream_.is_open()) {
        std::cerr << "[CanInterface] Failed to open file: " << file_path << std::endl;
    }
}

bool CanInterface::readFileLine() {

    if (!file_stream_.is_open()) return false;

    std::string line;
    // We assume update() is called frequently.
    // We read ONE line per update to simulate a stream.
    // In a real replay tool we'd respect timestamps.
    // But for simple integration testing, line-by-line is sufficient.
    
    if (std::getline(file_stream_, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> parts;
        
        while (iss >> token) {
            parts.push_back(token);
        }

        // Simple parser for .asc line:
        // 0.022530 1 A1 Rx d 8 00 00 ...
        // Index: 0=Time, 1=Chan, 2=ID, ... 5=DLC, 6+=Data
        
        if (parts.size() >= 7) {
            try {
                // Check if it's a data frame
                bool is_rx = false;
                for (const auto& p : parts) { 
                    if (p == "Rx") {
                        is_rx = true; 
                    }
                }
                
                if (is_rx) {
                    // Extract ID (Hex)
                    int id = std::stoi(
                        parts[2], 
                        nullptr, 
                        16
                    );
                    
                    // Extract data
                    std::vector<uint8_t> data;
                    // Find where data starts (after 'd' and '8')
                    // Usually data starts at index 6 or 7 depending on format variation
                    // We scan from end or look for hex bytes
                    
                    int dlc_idx = -1;
                    for(size_t i = 0; i < parts.size(); ++i) {
                        if (parts[i] == "d") {
                            dlc_idx = i + 1; // Next is DLC length
                            break;
                        }
                    }

                    if (
                        dlc_idx != -1 && 
                        dlc_idx + 1 < (int)parts.size()
                    ) {
                        int dlc = std::stoi(parts[dlc_idx]);
                        for (int i = 0; i < dlc; ++i) {
                            if (dlc_idx + 1 + i < (int)parts.size()) {
                                data.push_back(std::stoi(
                                    parts[dlc_idx + 1 + i], 
                                    nullptr, 
                                    16
                                ));
                            }
                        }
                        parseFrame(id, data);
                        return true;
                    }
                }
            } catch (...) {
                // Ignore parsing errors for comments/header lines
            }
        }
        return true; // Line read successfully (even if empty/comment)
    } 
    
    return false; // End of file
}

// definition of global pointer
std::unique_ptr<CanInterface> can_interface=nullptr;

} // namespace autoware_pov::drivers