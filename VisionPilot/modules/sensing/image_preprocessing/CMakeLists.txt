cmake_minimum_required(VERSION 3.22.1)

find_package(OpenCV REQUIRED)

# ─── Optional VPI (NVIDIA Jetson Vision Programming Interface) ─────────
# Install: sudo apt install libvpi-dev
# VPI replaces cv::warpPerspective with GPU/VIC hardware-accelerated warp.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(VPI QUIET vpi)
endif()

if(VPI_FOUND)
    message(STATUS "VPI found — GPU-accelerated warp enabled")
    add_compile_definitions(VPI_AVAILABLE)
else()
    message(STATUS "VPI not found — using OpenCV CPU warp")
endif()

add_library(image_preprocessing
    src/image_preprocessor.cpp
    src/vpi_warp.cpp
)

target_include_directories(image_preprocessing
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(image_preprocessing
    PUBLIC
    common
    ${OpenCV_LIBS}
)

if(VPI_FOUND)
    target_include_directories(image_preprocessing
        SYSTEM PRIVATE ${VPI_INCLUDE_DIRS}
    )
    target_link_libraries(image_preprocessing
        PRIVATE ${VPI_LIBRARIES}
        PRIVATE cuda
    )
endif()