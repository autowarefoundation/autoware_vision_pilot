cmake_minimum_required(VERSION 3.22.1)

add_library(output_rtsp_stream
        src/output_rtsp_stream.cpp
)

target_include_directories(output_rtsp_stream
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)