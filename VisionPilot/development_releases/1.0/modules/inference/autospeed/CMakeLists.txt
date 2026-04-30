cmake_minimum_required(VERSION 3.22.1)

add_library(autospeed
        src/autospeed.cpp
)

target_include_directories(autospeed
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)