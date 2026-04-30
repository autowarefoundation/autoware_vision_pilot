cmake_minimum_required(VERSION 3.22.1)

add_library(autodrive
        src/autodrive.cpp
)

target_include_directories(autodrive
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)