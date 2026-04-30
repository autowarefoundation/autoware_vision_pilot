cmake_minimum_required(VERSION 3.22.1)

add_library(input_rtsp_stream
        src/input_rtsp_stream.cpp
)

target_include_directories(input_rtsp_stream
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)