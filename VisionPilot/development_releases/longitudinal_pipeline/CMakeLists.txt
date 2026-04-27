cmake_minimum_required(VERSION 3.16)
project(visionpilot_autospeed VERSION 0.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime - user must set ONNXRUNTIME_ROOT
if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT environment variable to ONNX Runtime installation directory\n"
                        "Example: export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.22.0")
endif()

set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)

# Find library (handle versioned .so files)
file(GLOB ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so*")
list(GET ONNXRUNTIME_LIBRARY 0 ONNXRUNTIME_LIBRARY)

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "ONNX Runtime headers not found at: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIB_DIR}")
endif()

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")

# AutoSpeed 2.0 inference backend (ONNX Runtime - Longitudinal)
add_library(autospeed_inference
    src/inference/autospeed/onnxruntime_session.cpp
    src/inference/autospeed/onnxruntime_engine.cpp
)
target_compile_definitions(autospeed_inference PRIVATE LOG_TYPE=0)
target_link_libraries(autospeed_inference
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
set_target_properties(autospeed_inference PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
    INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)

# Visualization module (longitudinal)
add_library(visionpilot_visualization_long
    src/visualization/visualize_long.cpp
)
target_compile_definitions(visionpilot_visualization_long PRIVATE LOG_TYPE=0)
target_link_libraries(visionpilot_visualization_long
    ${OpenCV_LIBS}
)

# Camera utilities module
add_library(camera_utils
    src/camera/camera_utils.cpp
)
target_link_libraries(camera_utils
    ${OpenCV_LIBS}
)

# Camera calibration module
add_library(camera_calibration
    src/camera/camera_calibration.cpp
)
target_link_libraries(camera_calibration
    ${OpenCV_LIBS}
)

# CAN Drivers module
add_library(can_drivers
    src/drivers/can_interface.cpp
)
target_link_libraries(can_drivers
    # Uses standard Linux headers (socketcan)
)

# Longitudinal tracking modules (ObjectFinder + Kalman + CIPO)
add_library(longitudinal_tracking
    src/tracking/object_finder.cpp
    src/tracking/kalman_filter.cpp
    src/tracking/cipo_history.cpp
    src/tracking/cipo_utils.cpp
    src/tracking/tracking_utils.cpp
    src/tracking/feature_matching_utils.cpp
)
target_compile_definitions(longitudinal_tracking PRIVATE LOG_TYPE=0)
target_link_libraries(longitudinal_tracking
    ${OpenCV_LIBS}
    yaml-cpp
)

# Speed planning module (RSS safe-distance + ACC/FCW/AEB policy)
add_library(speed_planning
    src/speed_planning/speed_planning.cpp
)
target_include_directories(speed_planning PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Longitudinal control module (PID controller for speed tracking)
add_library(longitudinal_control
    src/longitudinal/pi_controller.cpp
)
target_include_directories(longitudinal_control PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# IPC shared-memory publisher (POSIX shm: longitudinal outputs)
add_library(visionpilot_publisher
    src/publisher/visionpilot_shared_state.cpp
)
target_include_directories(visionpilot_publisher PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(visionpilot_publisher rt)   # -lrt for shm_open/shm_unlink

# Config reader module (properties file format - no dependencies)
add_library(config_reader
    src/config/config_reader.cpp
)

# Main executable
file(COPY ${CMAKE_SOURCE_DIR}/weights
        DESTINATION ${CMAKE_BINARY_DIR})

add_executable(visionpilot_autospeed
    main.cpp
)
target_compile_definitions(visionpilot_autospeed PRIVATE
        VISIONPILOT_SHARE_DIR="/usr/share/visionpilot"
        LOG_TYPE=0
)

# Link libraries
set(VISIONPILOT_LIBS
    autospeed_inference
    visionpilot_visualization_long
    longitudinal_tracking
    speed_planning
    longitudinal_control
    visionpilot_publisher
    camera_utils
    camera_calibration
    can_drivers
    config_reader
    ${OpenCV_LIBS}
    pthread
)

target_link_libraries(visionpilot_autospeed
        ${VISIONPILOT_LIBS}
        ${ONNXRUNTIME_LIBRARY}
)

set_target_properties(visionpilot_autospeed PROPERTIES
        BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
        INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)

# Installation
install(TARGETS visionpilot_autospeed
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_SOURCE_DIR}/visionpilot.conf
        DESTINATION share/visionpilot/)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/weights/"
        DESTINATION "share/visionpilot/weights")

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "VisionPilot AutoSpeed 2.0 (Longitudinal-only) build configured")

# Include CPack and configure DEB-specific variables
set(CPACK_PACKAGE_VENDOR "Autoware Foundation")
set(CPACK_PACKAGE_CONTACT "")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "atanasko.mitrev@autoware.org, tran.huy@autoware.org, pranav.doma@autoware.org")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VisionPilot AutoSpeed 2.0 Longitudinal Control")
set(CPACK_SYSTEM_NAME "x86_64")
set(CPACK_GENERATOR "DEB")
include(CPack)
