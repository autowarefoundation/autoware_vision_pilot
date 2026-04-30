cmake_minimum_required(VERSION 3.22.1)

add_library(can_writer
        src/can_writer.cpp
)

target_include_directories(can_writer
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)