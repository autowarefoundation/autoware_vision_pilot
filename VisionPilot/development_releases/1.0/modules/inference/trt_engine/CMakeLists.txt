cmake_minimum_required(VERSION 3.22.1)

add_library(trt_engine
        src/trt_engine.cpp
)

target_include_directories(trt_engine
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)