cmake_minimum_required(VERSION 3.22.1)

add_library(autosteer
        src/autosteer.cpp
)

target_include_directories(autosteer
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)