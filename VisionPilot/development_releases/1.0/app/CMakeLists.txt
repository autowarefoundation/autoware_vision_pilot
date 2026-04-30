add_executable(VisionPilot VisionPilot.cpp)

target_link_libraries(VisionPilot
        PRIVATE
        can_reader
        can_writer
        input_rtsp_stream
        output_rtsp_stream
        autodrive
        autospeed
        autosteer
        visualization
)