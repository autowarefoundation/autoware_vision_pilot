#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI/CLI.hpp>
#include <zenoh.h>

using namespace cv; 
using namespace std; 

#define DEFAULT_KEYEXPR "video/raw"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app{"Zenoh video publisher example"};
    // Add options
    std::string input_video_path;
    app.add_option("video_path", input_video_path, "Path to the input video file")->required()->check(CLI::ExistingFile);
    std::string keyexpr = DEFAULT_KEYEXPR;
    app.add_option("-k,--key", keyexpr, "The key expression to publish to")->default_val(DEFAULT_KEYEXPR);
    std::string config_path = "";
    app.add_option("-c,--config", config_path, "The configuration file. Currently, this file must be a valid JSON5 or YAML file.")->check(CLI::ExistingFile);;
    CLI11_PARSE(app, argc, argv);

    try {
        // Initialize video capture
        cv::VideoCapture cap(input_video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video stream or file: " + input_video_path);
        }
        const double fps = cap.get(cv::CAP_PROP_FPS);
        // const double fps = 60;
        const auto frame_duration = std::chrono::nanoseconds(static_cast<long long>(1'000'000'000.0 / fps));
        std::cout << "Publishing video from '" << input_video_path << "' (" << fps << " FPS) to key '" << keyexpr << "'..." << std::endl;

        // Create Zenoh session
        z_owned_config_t config;
        z_owned_session_t s;
        z_config_default(&config);
        if (!config_path.empty()) {
            std::cout << "Loading Zenoh config from: " << config_path << std::endl;
            z_owned_config_t loaded_config;
            if (zc_config_from_file(&loaded_config, config_path.c_str()) < 0) {
                throw std::runtime_error("Error loading Zenoh config from file: " + config_path);
            }
            z_drop(z_move(config));
            config = loaded_config;
        }

        if (z_open(&s, z_move(config), NULL) < 0) {
            throw std::runtime_error("Error opening Zenoh session");
        }
        // Declare a Zenoh publisher
        z_owned_publisher_t pub;
        z_view_keyexpr_t ke;
        z_view_keyexpr_from_str(&ke, keyexpr.c_str());
        if (z_declare_publisher(z_loan(s), &pub, z_loan(ke), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh publisher for key expression: " + keyexpr);
        }

        // Display video frames
        cv::Mat frame;
        while (cap.read(frame)) {
            auto start_time = std::chrono::steady_clock::now();

            // Set Zenoh publisher options
            z_publisher_put_options_t options;
            z_publisher_put_options_default(&options);
            // Put the frame information into the attachment
            //std::cout << "Frame: dataSize=" << dataSize << ", row=" << frame.rows << ", cols=" << frame.cols << ", type=" << frame.type() << std::endl;
            z_owned_bytes_t attachment;
            // row, col, type
            int input_bytes[] = {frame.rows, frame.cols, frame.type()};
            z_bytes_copy_from_buf(&attachment, (const uint8_t*)input_bytes, sizeof(input_bytes));
            options.attachment = z_move(attachment);
            
            // Add timestamp
            z_timestamp_t ts;
            z_timestamp_new(&ts, z_loan(s));
            options.timestamp = &ts;

            // Publish images
            unsigned char* pixelPtr = frame.data; 
            size_t dataSize = frame.total() * frame.elemSize(); 
            z_owned_bytes_t payload;
            z_bytes_copy_from_buf(&payload, pixelPtr, dataSize);
            z_publisher_put(z_loan(pub), z_move(payload), &options);

            // Sleep for the remaining time to maintain the frame rate, accounting for processing time
            auto end_time = std::chrono::steady_clock::now();
            auto elapsed_time = end_time - start_time;
            if (elapsed_time < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed_time);
            }
        }
        
        // Clean up
        z_drop(z_move(pub));
        z_drop(z_move(s));
        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 