cmake_minimum_required(VERSION 3.22.1)

add_library(can_reader
        src/can_reader.cpp
)

target_include_directories(can_reader
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)