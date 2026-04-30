cmake_minimum_required(VERSION 3.22.1)

project(VisionPilot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put all executables in build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Modules
add_subdirectory(modules/can_reader)
add_subdirectory(modules/can_writer)
add_subdirectory(modules/input_rtsp_stream)
add_subdirectory(modules/output_rtsp_stream)
add_subdirectory(modules/visualization)

# Inference
add_subdirectory(modules/inference/trt_engine)
add_subdirectory(modules/inference/autodrive)
add_subdirectory(modules/inference/autospeed)
add_subdirectory(modules/inference/autosteer)

# App
add_subdirectory(app)
