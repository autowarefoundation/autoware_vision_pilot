#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <CLI/CLI.hpp>
#include <zenoh.h>

#include "autospeed/detection.hpp"
#include "object_finder.hpp"
#include "autospeed_visualization_engine.hpp"
#include "autospeed_payload.hpp"

using namespace cv;
using namespace std;
using namespace autoware_pov::vision;

#define DEFAULT_VIDEO_KEYEXPR "video/input"
#define DEFAULT_RESULT_KEYEXPR "video/output"

#define RECV_BUFFER_SIZE 100

z_owned_slice_t decode_frame_from_sample(const z_owned_sample_t& sample, int& row, int& col, int& type) {
    const z_loaned_sample_t* loaned_sample = z_loan(sample);
    z_owned_slice_t zslice;
    if (Z_OK != z_bytes_to_slice(z_sample_payload(loaned_sample), &zslice)) {
        throw std::runtime_error("Wrong payload");
    }

    // Extract the frame information for the attachment
    const z_loaned_bytes_t* attachment = z_sample_attachment(loaned_sample);
    if (attachment != NULL) {
        z_owned_slice_t output_bytes;
        int attachment_arg[3];
        z_bytes_to_slice(attachment, &output_bytes);
        memcpy(attachment_arg, z_slice_data(z_loan(output_bytes)), z_slice_len(z_loan(output_bytes)));
        row = attachment_arg[0];
        col = attachment_arg[1];
        type = attachment_arg[2];
        z_drop(z_move(output_bytes));
    } else {
        z_drop(z_move(zslice));
        throw std::runtime_error("No attachment");
    }

    return zslice;
}

// Parse unified AutoSpeed payload
void parseAutoSpeedPayload(const z_owned_sample_t& sample, zenoh::AutoSpeedPayload& payload) {
    const z_loaned_sample_t* loaned_sample = z_loan(sample);

    z_owned_slice_t zslice;
    if (Z_OK != z_bytes_to_slice(z_sample_payload(loaned_sample), &zslice)) {
        throw std::runtime_error("Wrong AutoSpeed payload");
    }

    memcpy(&payload, z_slice_data(z_loan(zslice)), sizeof(zenoh::AutoSpeedPayload));
    z_drop(z_move(zslice));
}

int main(int argc, char** argv) {
    // Parse command line arguments
    CLI::App app{"AutoSpeed detection visualizer"};
    std::string video_keyexpr = DEFAULT_VIDEO_KEYEXPR;
    app.add_option("-v,--video-key", video_keyexpr, "The key expression for video input")
        ->default_val(DEFAULT_VIDEO_KEYEXPR);
    std::string result_keyexpr = DEFAULT_RESULT_KEYEXPR;
    app.add_option("-k,--key", result_keyexpr, "The key expression for AutoSpeed results")
        ->default_val(DEFAULT_RESULT_KEYEXPR);
    std::string config_path = "";
    app.add_option("-c,--config", config_path, "The Zenoh configuration file")
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);

    try {
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

        // Declare video subscriber
        z_owned_subscriber_t video_sub;
        z_view_keyexpr_t video_ke;
        z_view_keyexpr_from_str(&video_ke, video_keyexpr.c_str());
        z_owned_ring_handler_sample_t video_handler;
        z_owned_closure_sample_t video_closure;
        z_ring_channel_sample_new(&video_closure, &video_handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &video_sub, z_loan(video_ke), z_move(video_closure), NULL) < 0) {
            throw std::runtime_error("Error declaring video subscriber: " + video_keyexpr);
        }

        // Declare result subscriber
        z_owned_subscriber_t result_sub;
        z_view_keyexpr_t result_ke;
        z_view_keyexpr_from_str(&result_ke, result_keyexpr.c_str());
        z_owned_ring_handler_sample_t result_handler;
        z_owned_closure_sample_t result_closure;
        z_ring_channel_sample_new(&result_closure, &result_handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &result_sub, z_loan(result_ke), z_move(result_closure), NULL) < 0) {
            throw std::runtime_error("Error declaring result subscriber: " + result_keyexpr);
        }

        std::cout << "Subscribing to video: '" << video_keyexpr << "'" << std::endl;
        std::cout << "Subscribing to results: '" << result_keyexpr << "'" << std::endl;
        std::cout << "Press ESC to stop." << std::endl;

        // Create visualization engine
        auto viz_engine = std::make_unique<autoware_pov::common::AutoSpeedVisualizationEngine>();

        z_owned_sample_t video_sample;
        zenoh::AutoSpeedPayload payload;

        while (Z_OK == z_recv(z_loan(video_handler), &video_sample)) {
            // Decode video frame
            int row, col, type;
            z_owned_slice_t video_slice = decode_frame_from_sample(video_sample, row, col, type);
            const uint8_t* video_ptr = z_slice_data(z_loan(video_slice));
            z_drop(z_move(video_sample));

            cv::Mat frame(row, col, type, const_cast<uint8_t*>(video_ptr));

            // Try to receive result data
            z_owned_sample_t result_sample;
            if (Z_OK == z_try_recv(z_loan(result_handler), &result_sample)) {
                parseAutoSpeedPayload(result_sample, payload);
                z_drop(z_move(result_sample));
            }

            // Extract detections from payload
            std::vector<autospeed::Detection> detections(
                payload.detections, payload.detections + payload.num_detections);

            // Convert TrackedObjectPayload to TrackedObject for visualization
            std::vector<TrackedObject> tracked_objects;
            tracked_objects.reserve(payload.num_tracked);
            for (int i = 0; i < payload.num_tracked; ++i) {
                const auto& tp = payload.tracked[i];
                TrackedObject obj;
                obj.track_id = tp.track_id;
                obj.class_id = tp.class_id;
                obj.confidence = tp.confidence;
                obj.bbox = cv::Rect(tp.bbox[0], tp.bbox[1], tp.bbox[2], tp.bbox[3]);
                obj.distance_m = tp.distance_m;
                obj.velocity_ms = tp.velocity_ms;
                tracked_objects.push_back(obj);
            }

            // Visualize using the engine (handles waitKey internally)
            viz_engine->visualize(frame, detections, tracked_objects,
                                  payload.cipo, payload.cut_in_detected, payload.kalman_reset);

            z_drop(z_move(video_slice));
        }

        // Cleanup
        z_drop(z_move(video_handler));
        z_drop(z_move(video_sub));
        z_drop(z_move(result_handler));
        z_drop(z_move(result_sub));
        z_drop(z_move(s));
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
