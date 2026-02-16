cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(video_pubsub)

# Don't remove RPATH while install binaries
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

## Find libraries
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)
find_package(zenohc REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

## ONNX Runtime Providers
option(LIBPNG_ROOTDIR "libpng root dir")
set(CMAKE_CXX_STANDARD 17)

# set the MODEL_ROOTDIR variable to the path where the model is located
set(MODEL_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/../../common")

include_directories(
                     ${OpenCV_INCLUDE_DIRS}    # OpenCV
                     ${MODEL_ROOTDIR}/include  # Common libraries
                     ${MODEL_ROOTDIR}/backends # Backend types (autospeed)
                     ${CMAKE_CURRENT_SOURCE_DIR}/../models  # Zenoh payload types
                   )

## Build & Link
add_executable(video_publisher video_publisher.cpp)
target_link_libraries(video_publisher PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
add_executable(video_subscriber video_subscriber.cpp)
target_link_libraries(video_subscriber PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
add_executable(segment_subscriber segment_subscriber.cpp
               ${MODEL_ROOTDIR}/visualizers/masks_visualization_engine.cpp)
target_link_libraries(segment_subscriber PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
add_executable(depth_subscriber depth_subscriber.cpp
               ${MODEL_ROOTDIR}/visualizers/depth_visualization_engine.cpp)
target_link_libraries(depth_subscriber PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)

# AutoSpeed detection subscriber
add_executable(autospeed_subscriber autospeed_subscriber.cpp
               ${MODEL_ROOTDIR}/visualizers/autospeed_visualization_engine.cpp
               ${MODEL_ROOTDIR}/tracking/kalman_filter.cpp)
target_link_libraries(autospeed_subscriber PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)

# Install
install(TARGETS video_publisher DESTINATION bin)
install(TARGETS video_subscriber DESTINATION bin)
install(TARGETS segment_subscriber DESTINATION bin)
install(TARGETS depth_subscriber DESTINATION bin)
install(TARGETS autospeed_subscriber DESTINATION bin)
