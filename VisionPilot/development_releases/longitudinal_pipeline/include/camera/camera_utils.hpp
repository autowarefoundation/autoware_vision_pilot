/**
 * @file camera_utils.hpp
 * @brief Simple camera device detection and management utilities
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace autoware_pov::vision::camera {

/**
 * @brief Camera device information
 */
struct CameraDevice {
    std::string path;        // /dev/video0 (standard V4L2 device path)
    std::string name;        // Camera name from driver
    int width = 0;
    int height = 0;
    double fps = 0.0;
};

/**
 * @brief List all /dev/video* devices (standard V4L2 location)
 * @return Vector of detected camera devices
 */
std::vector<CameraDevice> listCameras();

/**
 * @brief Check if driver package exists in camera_driver/ folder
 * @return true if .deb file found in camera_driver/
 */
bool checkDriverPackage();

/**
 * @brief Install driver from camera_driver/ folder
 * @return true if installation succeeded
 */
bool installDriver();

/**
 * @brief Interactive device selection with installation loop
 * @return Selected device path, or empty string if none/cancelled
 */
std::string selectCamera();

/**
 * @brief Verify camera is accessible and can capture frames
 * @param device_path Path to camera device (e.g., /dev/video0)
 * @return true if camera is working
 */
bool verifyCamera(const std::string& device_path);

/**
 * @brief Open camera with auto-configuration
 * @param device_path Path to camera device
 * @return Configured cv::VideoCapture object
 */
cv::VideoCapture openCamera(const std::string& device_path);

/**
 * @brief Print manual installation instructions
 */
void printDriverInstructions();

} // namespace autoware_pov::vision::camera

