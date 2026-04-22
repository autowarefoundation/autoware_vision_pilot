/**
 * @file camera_utils.cpp
 * @brief Implementation of camera device detection and management
 */

#include "camera/camera_utils.hpp"
#include <iostream>
#include <filesystem>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

namespace fs = std::filesystem;

namespace autoware_pov::vision::camera {

std::vector<CameraDevice> listCameras()
{
    std::vector<CameraDevice> devices;
    
    // Scan all /dev/video* devices (standard V4L2 location)
    for (int i = 0; i < 10; ++i) {
        std::string path = "/dev/video" + std::to_string(i);
        
        if (!fs::exists(path)) continue;
        
        int fd = open(path.c_str(), O_RDWR);
        if (fd < 0) continue;
        
        v4l2_capability cap;
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            // Only video capture devices
            if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
                CameraDevice dev;
                dev.path = path;
                dev.name = std::string(reinterpret_cast<char*>(cap.card));
                
                // Get current format
                v4l2_format fmt;
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (ioctl(fd, VIDIOC_G_FMT, &fmt) == 0) {
                    dev.width = fmt.fmt.pix.width;
                    dev.height = fmt.fmt.pix.height;
                }
                
                // Get frame rate
                v4l2_streamparm parm;
                parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (ioctl(fd, VIDIOC_G_PARM, &parm) == 0) {
                    if (parm.parm.capture.timeperframe.denominator > 0) {
                        dev.fps = static_cast<double>(parm.parm.capture.timeperframe.denominator) / 
                                  parm.parm.capture.timeperframe.numerator;
                    }
                }
                
                devices.push_back(dev);
            }
        }
        close(fd);
    }
    
    return devices;
}

bool checkDriverPackage()
{
    // Check if camera_driver/ folder exists and has .deb files
    if (!fs::exists("camera_driver")) {
        return false;
    }
    
    try {
        for (const auto& entry : fs::directory_iterator("camera_driver")) {
            if (entry.path().extension() == ".deb") {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    
    return false;
}

std::string findDriverPackage()
{
    if (!fs::exists("camera_driver")) {
        return "";
    }
    
    try {
        for (const auto& entry : fs::directory_iterator("camera_driver")) {
            if (entry.path().extension() == ".deb") {
                return entry.path().string();
            }
        }
    } catch (...) {
        return "";
    }
    
    return "";
}

bool installDriver()
{
    std::string deb_file = findDriverPackage();
    
    if (deb_file.empty()) {
        std::cerr << "No .deb package found in camera_driver/ folder" << std::endl;
        return false;
    }
    
    std::cout << "Found driver package: " << deb_file << std::endl;
    std::cout << "Installing..." << std::endl;
    
    // Install the package
    std::string install_cmd = "sudo dpkg -i " + deb_file;
    int ret = system(install_cmd.c_str());
    
    if (ret != 0) {
        std::cerr << "Installation failed" << std::endl;
        return false;
    }
    
    std::cout << "Package installed. Loading kernel modules..." << std::endl;
    
    // Try to load modules (the .deb should have set up autoload, but try anyway)
    // This will fail silently if already loaded or if module name is different
    system("sudo modprobe -a 2>/dev/null || true");
    
    // Give system time to create devices
    std::cout << "Waiting for devices to initialize..." << std::endl;
    sleep(2);
    
    return true;
}

std::string selectCamera()
{
    auto devices = listCameras();
    
    // Loop: check devices → install if needed → re-check → exit if still none
    int attempts = 0;
    const int max_attempts = 2;
    
    while (devices.empty() && attempts < max_attempts) {
        std::cout << "\n❌ No camera devices found.\n" << std::endl;
        
        // Check if we have driver package
        if (checkDriverPackage()) {
            std::cout << "Found driver package in camera_driver/ folder." << std::endl;
            std::cout << "Would you like to install it? (y/n): ";
            
            std::string response;
            std::getline(std::cin, response);
            
            if (response == "y" || response == "Y") {
                if (installDriver()) {
                    std::cout << "✓ Driver installed. Checking for devices...\n" << std::endl;
                    devices = listCameras();
                    attempts++;
                } else {
                    std::cerr << "\nDriver installation failed." << std::endl;
                    break;
                }
            } else {
                break;
            }
        } else {
            std::cout << "No driver package found in camera_driver/ folder." << std::endl;
            std::cout << "Please place your camera driver (.deb file) in camera_driver/ folder." << std::endl;
            break;
        }
    }
    
    // Still no devices after attempts
    if (devices.empty()) {
        std::cout << "\nExiting. Please install driver manually and try again." << std::endl;
        return "";
    }
    
    // Show available devices
    std::cout << "\n📷 Available cameras:\n" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i].path 
                  << " - " << devices[i].name;
        
        if (devices[i].width > 0 && devices[i].height > 0) {
            std::cout << " (" << devices[i].width << "x" << devices[i].height;
            if (devices[i].fps > 0) {
                std::cout << " @ " << devices[i].fps << " fps";
            }
            std::cout << ")";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nSelect camera [0-" << (devices.size() - 1) << "] or 'q' to quit: ";
    std::string input;
    std::getline(std::cin, input);
    
    if (input == "q" || input == "Q") {
        return "";
    }
    
    try {
        int idx = std::stoi(input);
        if (idx >= 0 && idx < static_cast<int>(devices.size())) {
            return devices[idx].path;
        }
    } catch (...) {
        std::cerr << "Invalid selection." << std::endl;
        return "";
    }
    
    std::cerr << "Invalid selection." << std::endl;
    return "";
}

bool verifyCamera(const std::string& device_path)
{
    std::cout << "Verifying camera: " << device_path << "..." << std::endl;
    
    cv::VideoCapture cap(device_path, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera." << std::endl;
        return false;
    }
    
    cv::Mat test_frame;
    if (!cap.read(test_frame) || test_frame.empty()) {
        std::cerr << "Cannot capture frame." << std::endl;
        cap.release();
        return false;
    }
    
    std::cout << "✓ Camera working!" << std::endl;
    cap.release();
    return true;
}

cv::VideoCapture openCamera(const std::string& device_path)
{
    cv::VideoCapture cap(device_path, cv::CAP_V4L2);
    
    // Let driver handle all settings automatically
    // OpenCV will query and use the device's default format
    
    return cap;
}

void printDriverInstructions()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Manual Driver Installation" << std::endl;
    std::cout << "========================================\n" << std::endl;
    std::cout << "1. Place driver package (.deb) in camera_driver/ folder\n" << std::endl;
    std::cout << "2. Install manually:" << std::endl;
    std::cout << "   sudo dpkg -i camera_driver/*.deb" << std::endl;
    std::cout << "   sudo modprobe -a\n" << std::endl;
    std::cout << "3. Verify devices:" << std::endl;
    std::cout << "   ls -l /dev/video*\n" << std::endl;
    std::cout << "4. Run this program again" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

} // namespace autoware_pov::vision::camera

