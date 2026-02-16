cmake_minimum_required(VERSION 3.16)
project(visionpilot VERSION 0.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

 # Find dependencies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Config reader uses properties file format (no external dependencies needed)

# Option to skip ONNX Runtime and use TensorRT directly
option(SKIP_ORT "Skip ONNX Runtime and use TensorRT directly" OFF)

if(SKIP_ORT)
    message(STATUS "SKIP_ORT=ON: Using TensorRT directly (no ONNX Runtime)")
    # Propagate SKIP_ORT macro to all targets
    add_compile_definitions(SKIP_ORT=1)
    
    # Find CUDA
    find_package(CUDA REQUIRED)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found. TensorRT requires CUDA.")
    endif()
    
    # Find TensorRT
    # Try common locations
    set(TENSORRT_SEARCH_PATHS
        /usr/local/cuda
        /usr/local/cuda-12.0
        /usr/local/cuda-11.8
        /usr/local/TensorRT-8.6.1.6
        /usr/local/TensorRT-8.5.3.1
        $ENV{TENSORRT_ROOT}
        $ENV{CUDA_HOME}
    )
    
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS ${TENSORRT_SEARCH_PATHS}
        PATH_SUFFIXES include
    )
    
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS ${TENSORRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64 lib/x64
    )
    
    find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
        PATHS ${TENSORRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64 lib/x64
    )
    
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_PARSER_LIBRARY)
        message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or install TensorRT.\n"
                            "Expected: TENSORRT_INCLUDE_DIR, TENSORRT_LIBRARY, TENSORRT_ONNX_PARSER_LIBRARY")
    endif()
    
    message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  - Library: ${TENSORRT_LIBRARY}")
    message(STATUS "  - ONNX Parser: ${TENSORRT_ONNX_PARSER_LIBRARY}")
    message(STATUS "CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
    
    # Include directories
    include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )
    
    # EgoLanes inference backend (TensorRT)
    add_library(visionpilot_inference
        src/inference/tensorrt_engine.cpp
        src/inference/tensorrt_autosteer_engine.cpp
    )
    target_compile_definitions(visionpilot_inference PRIVATE LOG_TYPE=0 SKIP_ORT=1)
    target_link_libraries(visionpilot_inference
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
    
else()
    message(STATUS "SKIP_ORT=OFF: Using ONNX Runtime")
    
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

    # EgoLanes inference backend (ONNX Runtime)
    add_library(visionpilot_inference
        src/inference/onnxruntime_session.cpp
        src/inference/onnxruntime_engine.cpp
        src/inference/autosteer_engine.cpp
    )
    target_compile_definitions(visionpilot_inference PRIVATE LOG_TYPE=0)
    target_link_libraries(visionpilot_inference
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARY}
    )
    # Add RPATH so executable can find libonnxruntime.so at runtime
    set_target_properties(visionpilot_inference PROPERTIES
        BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
        INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
    )
    
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
    message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
    message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# Visualization module
add_library(visionpilot_visualization
    src/visualization/draw_lanes.cpp
)
target_compile_definitions(visionpilot_visualization PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(visionpilot_visualization
    visionpilot_inference
    ${OpenCV_LIBS}
)

# Lane filtering module
add_library(visionpilot_lane_filtering
    src/lane_filtering/lane_filter.cpp
)
target_link_libraries(visionpilot_lane_filtering
    visionpilot_inference
    ${OpenCV_LIBS}
)

# Camera utilities module
add_library(camera_utils
    src/camera/camera_utils.cpp
)
target_link_libraries(camera_utils
    ${OpenCV_LIBS}
)

# Path planning module (polynomial fitting + Bayes filter)
add_library(path_planning
    src/path_planning/estimator.cpp
    src/path_planning/poly_fit.cpp
    src/path_planning/path_finder.cpp
)
target_link_libraries(path_planning
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Steering control module
add_library(steering_control
    src/steering_control/steering_controller.cpp
        src/steering_control/steering_filter.cpp
        include/steering_control/steering_filter.hpp
)
target_link_libraries(steering_control
    # No external dependencies, pure C++ math
)

# CAN Drivers module
add_library(can_drivers
    src/drivers/can_interface.cpp
)
target_link_libraries(can_drivers
    # Uses standard Linux headers (socketcan)
)

# Publish steering and velocity
add_library(publisher
        include/publisher/ipc_shared_state.hpp
        src/publisher/ipc_shared_state.cpp
)
target_link_libraries(publisher
        # No external dependencies, pure C++ math
)

# Rerun SDK (optional, for visualization/logging)
option(ENABLE_RERUN "Enable Rerun logging and visualization" OFF)

if(ENABLE_RERUN)
    if(NOT DEFINED ENV{RERUN_SDK_ROOT})
        message(WARNING "RERUN_SDK_ROOT not set. Disabling Rerun support.\n"
                        "To enable: export RERUN_SDK_ROOT=/path/to/rerun_cpp_sdk")
        set(ENABLE_RERUN OFF)
    else()
        set(RERUN_SDK_ROOT $ENV{RERUN_SDK_ROOT})

        # Check if Rerun SDK exists (src/rerun.hpp for SDK bundle)
        if(NOT EXISTS ${RERUN_SDK_ROOT}/src/rerun.hpp)
            message(WARNING "Rerun SDK not found at ${RERUN_SDK_ROOT}. Disabling Rerun support.")
            set(ENABLE_RERUN OFF)
        else()
            message(STATUS "Rerun SDK: ${RERUN_SDK_ROOT}")

            # Configure Rerun to use system-installed Arrow (no download)
            set(RERUN_DOWNLOAD_AND_BUILD_ARROW OFF CACHE BOOL "Use system Arrow" FORCE)

            # Find Arrow (should be at /usr/local after manual install)
            set(CMAKE_PREFIX_PATH "/usr/local;${CMAKE_PREFIX_PATH}")
            find_package(Arrow REQUIRED)
            message(STATUS "  - Arrow found: ${Arrow_DIR}")

            # Add Rerun SDK as subdirectory (will build rerun_cpp from source)
            add_subdirectory(${RERUN_SDK_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/rerun_sdk_build EXCLUDE_FROM_ALL)

            # Rerun logger module
            add_library(rerun_logger
                src/rerun/rerun_logger.cpp
            )
            target_compile_definitions(rerun_logger PRIVATE ENABLE_RERUN)
target_link_libraries(rerun_logger
    visionpilot_inference
                ${OpenCV_LIBS}
                rerun_sdk
            )

            set(RERUN_ENABLED TRUE)
            message(STATUS "Rerun support: ENABLED")
        endif()
    endif()
endif()

if(NOT RERUN_ENABLED)
    message(STATUS "Rerun support: DISABLED")
endif()

# Lane tracking module
add_library(visionpilot_lane_tracking
    src/lane_tracking/lane_tracking.cpp
)
target_link_libraries(visionpilot_lane_tracking
    visionpilot_inference
    ${OpenCV_LIBS}
)

# Config reader module (properties file format - no dependencies)
add_library(config_reader
    src/config/config_reader.cpp
)

# Main executable
file(COPY ${CMAKE_SOURCE_DIR}/images
        DESTINATION ${CMAKE_BINARY_DIR})

add_executable(visionpilot
    main.cpp
)
target_compile_definitions(visionpilot PRIVATE
        VISIONPILOT_SHARE_DIR="/usr/share/visionpilot"
        LOG_TYPE=0
)  # Use non-ROS logging

# Link libraries
set(VISIONPILOT_LIBS
    visionpilot_inference
    visionpilot_visualization
    visionpilot_lane_filtering
    visionpilot_lane_tracking
    camera_utils
    path_planning
    steering_control
    can_drivers
    config_reader
    publisher
    ${OpenCV_LIBS}
    pthread
)

if(ENABLE_RERUN AND RERUN_ENABLED)
    target_compile_definitions(visionpilot PRIVATE ENABLE_RERUN)
    list(APPEND VISIONPILOT_LIBS rerun_logger)
endif()

if(NOT SKIP_ORT)
    target_link_libraries(visionpilot
            ${VISIONPILOT_LIBS}
            ${ONNXRUNTIME_LIBRARY}
    )

    set_target_properties(visionpilot PROPERTIES
            BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
            INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
    )
else()
    target_link_libraries(visionpilot
            ${VISIONPILOT_LIBS}
    )
endif()

# Test executable for AutoSteer (ONNX Runtime only)
if(NOT SKIP_ORT)
    add_executable(test_autosteer
        test/test_autosteer.cpp
    )
    target_compile_definitions(test_autosteer PRIVATE LOG_TYPE=0)
    target_link_libraries(test_autosteer ${VISIONPILOT_LIBS})
endif()

# Installation
install(TARGETS visionpilot
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_SOURCE_DIR}/visionpilot.conf
        DESTINATION share/visionpilot/)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/images/"
        DESTINATION "share/visionpilot/images")

install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
        DESTINATION "share/visionpilot/models"
        FILES_MATCHING
        PATTERN "*.onnx"
        PATTERN "trt_cache" EXCLUDE)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/onnxruntime/"
        DESTINATION "share/visionpilot/onnxruntime")

message(STATUS "OpenCV: ${OpenCV_VERSION}")

# Include CPack and configure DEB-specific variables
set(CPACK_PACKAGE_VENDOR "Autoware Foundation")
set(CPACK_PACKAGE_CONTACT "")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "atanasko.mitrev@autoware.org, tran.huy@autoware.org, pranav.doma@autoware.org")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VisionPilot v0.5")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
        "cuda-toolkit-12-6,
        libcudnn9-cuda-12,
        libcublaslt11,
        libcudart11.0,
        libnvinfer10,
        libnvinfer-plugin10,
        libnvonnxparsers10,
        cuda-drivers"
)

set(CPACK_SYSTEM_NAME "x86_64")
set(CPACK_GENERATOR "DEB")
include(CPack)


