#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <cstring>

#include <CLI/CLI.hpp>
#include <zenoh.h>

#include "inference_backend_base.hpp"
#include "onnx_runtime_backend.hpp"
#include "tensorrt_backend.hpp"
#include "masks_visualization_engine.hpp"
#include "masks_visualization_kernels.hpp"
#include "depth_visualization_engine.hpp"
#include "fps_timer.hpp"

// AutoSpeed includes
#include "autospeed/tensorrt_engine.hpp"
#include "autospeed/onnxruntime_engine.hpp"
#include "object_finder.hpp"
#include "autospeed_payload.hpp"

using namespace cv; 
using namespace std; 

using namespace autoware_pov::vision;

#define VIDEO_INPUT_KEYEXPR "video/input"
#define VIDEO_OUTPUT_KEYEXPR "video/output"
#define DEFAULT_BACKEND "onnxruntime"
#define DEFAULT_PRECISION "cuda"
#define DEFAULT_GPU_ID 0
#define MODEL_TYPE "segmentation"

#define BENCHMARK_OUTPUT_FREQUENCY 100
#define RECV_BUFFER_SIZE 100

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app{"Zenoh video scene segmentation visualizer"};
    std::string model_path;
    // Add options
    app.add_option("model_path", model_path, "Path to the ONNX model file")->required()->check(CLI::ExistingFile);
    std::string input_keyexpr = VIDEO_INPUT_KEYEXPR;
    app.add_option("-i,--input-key", input_keyexpr, "The key expression to subscribe video from")
        ->default_val(VIDEO_INPUT_KEYEXPR);
    std::string output_keyexpr = VIDEO_OUTPUT_KEYEXPR;
    app.add_option("-o,--output-key", output_keyexpr, "The key expression to publish the result to")
        ->default_val(VIDEO_OUTPUT_KEYEXPR);
    std::string backend = DEFAULT_BACKEND;
    app.add_option("-b,--backend", backend, "Inference backend to use (onnxruntime or tensorrt)")
        ->default_val(DEFAULT_BACKEND);
    std::string precision = DEFAULT_PRECISION;
    app.add_option("-p,--precision", precision, "Precision for the backend (cpu, cuda, fp16, fp32)")
        ->default_val(DEFAULT_PRECISION);
    int gpu_id = DEFAULT_GPU_ID;
    app.add_option("-g,--gpu-id", gpu_id, "GPU ID to use for CUDA backend")
        ->default_val(DEFAULT_GPU_ID);
    std::string model_type = "scene";
    app.add_option("-m,--model-type", model_type, "Type of the model (segmentation, depth, or autospeed)")
        ->default_val("segmentation");
    std::string config_path = "";
    app.add_option("-c,--config", config_path, "The configuration file. Currently, this file must be a valid JSON5 or YAML file.")
        ->check(CLI::ExistingFile);

    // AutoSpeed-specific options
    std::string homography_path = "";
    app.add_option("--homography", homography_path, "Homography YAML file for distance estimation (autospeed only)");
    float conf_threshold = 0.6f;
    app.add_option("--conf", conf_threshold, "Confidence threshold (autospeed only)")
        ->default_val(0.6f);
    float iou_threshold = 0.45f;
    app.add_option("--iou", iou_threshold, "NMS IoU threshold (autospeed only)")
        ->default_val(0.45f);

    CLI11_PARSE(app, argc, argv);

    try {
        // Initialize inference engine based on model type
        std::unique_ptr<InferenceBackend> backend_;
        std::unique_ptr<autospeed::AutoSpeedTensorRTEngine> autospeed_trt_backend_;
        std::unique_ptr<autospeed::AutoSpeedOnnxEngine> autospeed_onnx_backend_;
        std::unique_ptr<ObjectFinder> object_finder_;

        if (model_type == "autospeed") {
            // AutoSpeed mode: use specialized engines
            if (backend == "tensorrt") {
                autospeed_trt_backend_ = std::make_unique<autospeed::AutoSpeedTensorRTEngine>(
                    model_path, precision, gpu_id
                );
                std::cout << "Initialized AutoSpeed TensorRT engine" << std::endl;
            } else if (backend == "onnxruntime") {
                autospeed_onnx_backend_ = std::make_unique<autospeed::AutoSpeedOnnxEngine>(
                    model_path, precision, precision, gpu_id
                );
                std::cout << "Initialized AutoSpeed ONNX Runtime engine" << std::endl;
            } else {
                throw std::invalid_argument("Unknown backend type for autospeed.");
            }
            // ObjectFinder will be lazily initialized after receiving first frame
            // (needs actual image dimensions from Zenoh attachment)
            if (homography_path.empty()) {
                std::cout << "Warning: No homography file provided, tracking will not calculate distances" << std::endl;
            }
        } else {
            // Segmentation/Depth mode: use generic backends
            if (backend == "onnxruntime") {
                backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
            } else if (backend == "tensorrt") {
                backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
            } else {
                throw std::invalid_argument("Unknown backend type.");
            }
        }

        // Zenoh Initialization
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
        // Declare a Zenoh subscriber
        z_owned_subscriber_t sub;
        z_view_keyexpr_t in_ke;
        z_view_keyexpr_from_str(&in_ke, input_keyexpr.c_str());
        z_owned_ring_handler_sample_t handler;
        z_owned_closure_sample_t closure;
        z_ring_channel_sample_new(&closure, &handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &sub, z_loan(in_ke), z_move(closure), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh subscriber for key expression: " + input_keyexpr);
        }
        // Declare a Zenoh publisher for the output
        z_owned_publisher_t pub;
        z_view_keyexpr_t out_ke;
        z_view_keyexpr_from_str(&out_ke, output_keyexpr.c_str());
        if (z_declare_publisher(z_loan(s), &pub, z_loan(out_ke), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh publisher for key expression: " + output_keyexpr);
        }

        // Subscribe to the input key expression and process frames
        std::cout << "Subscribing to '" << input_keyexpr << "'..." << std::endl;
        std::cout << "Publishing results to '" << output_keyexpr << "'..." << std::endl;
        z_owned_sample_t sample;

        // Benchmark: Output results at a certain frequency
        FpsTimer timer(BENCHMARK_OUTPUT_FREQUENCY);

        while (Z_OK == z_recv(z_loan(handler), &sample)) {
            // Benchmark: Receive new frame
            timer.startNewFrame();

            // Get the loaned sample and extract the payload
            const z_loaned_sample_t* loaned_sample = z_loan(sample);
            z_owned_slice_t zslice;
            if (Z_OK != z_bytes_to_slice(z_sample_payload(loaned_sample), &zslice)) {
                throw std::runtime_error("Wrong payload");
            }
            const uint8_t* ptr = z_slice_data(z_loan(zslice));
            // Extract the frame information for the attachment
            const z_loaned_bytes_t* attachment = z_sample_attachment(loaned_sample);
            int row, col, type;
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
                throw std::runtime_error("No attachment");
            }

            cv::Mat frame(row, col, type, (uint8_t *)ptr);

            // Benchmark: Preprocess done
            timer.recordPreprocessEnd();

            // AutoSpeed mode: different processing pipeline
            if (model_type == "autospeed") {
                // Lazy initialize ObjectFinder on first frame (now we have actual dimensions)
                if (!object_finder_ && !homography_path.empty()) {
                    object_finder_ = std::make_unique<ObjectFinder>(
                        homography_path, col, row  // col = width, row = height
                    );
                    std::cout << "Initialized ObjectFinder with homography: " << homography_path
                              << " (image size: " << col << "x" << row << ")" << std::endl;
                }

                // Run inference using AutoSpeed engine
                std::vector<autospeed::Detection> detections;
                if (autospeed_trt_backend_) {
                    detections = autospeed_trt_backend_->inference(frame, conf_threshold, iou_threshold);
                } else if (autospeed_onnx_backend_) {
                    detections = autospeed_onnx_backend_->inference(frame, conf_threshold, iou_threshold);
                }

                // Update tracking and get CIPO
                std::vector<TrackedObject> tracked_objects;
                CIPOInfo cipo;
                if (object_finder_) {
                    tracked_objects = object_finder_->update(detections, frame);
                    cipo = object_finder_->getCIPO(frame);
                }

                // Benchmark: Inference done
                timer.recordInferenceEnd();

                // Build unified AutoSpeedPayload
                zenoh::AutoSpeedPayload payload;

                // Copy detections (POD, use memcpy)
                payload.num_detections = std::min(static_cast<int>(detections.size()),
                                                   zenoh::MAX_DETECTIONS);
                std::memcpy(payload.detections, detections.data(),
                            payload.num_detections * sizeof(autospeed::Detection));

                // Copy tracked objects (need conversion from TrackedObject)
                payload.num_tracked = std::min(static_cast<int>(tracked_objects.size()),
                                                zenoh::MAX_TRACKED);
                for (int i = 0; i < payload.num_tracked; ++i) {
                    payload.tracked[i] = zenoh::TrackedObjectPayload(tracked_objects[i]);
                }

                // Copy CIPO and event flags
                payload.cipo = cipo;
                if (object_finder_) {
                    payload.cut_in_detected = object_finder_->wasCutInDetected();
                    payload.kalman_reset = object_finder_->wasKalmanReset();
                    object_finder_->clearEventFlags();
                }

                // Publish unified payload
                z_publisher_put_options_t options;
                z_publisher_put_options_default(&options);
                z_owned_bytes_t payload_out;
                z_bytes_copy_from_buf(&payload_out,
                    reinterpret_cast<const uint8_t*>(&payload), sizeof(payload));
                z_publisher_put(z_loan(pub), z_move(payload_out), &options);

                // Benchmark: Output done
                timer.recordOutputEnd();

                // Release slice and sample
                z_drop(z_move(zslice));
                z_drop(z_move(sample));
                continue;  // Skip the rest of the loop for autospeed
            }

            // Segmentation/Depth mode: run inference using generic backend
            if (!backend_->doInference(frame)) {
                throw std::runtime_error("Failed to run inference on the frame");
            }

            // Model-type specific processing
            const float* tensor_data = backend_->getRawTensorData();
            std::vector<int64_t> tensor_shape = backend_->getTensorShape();
            if (tensor_shape.size() != 4) {
                throw std::runtime_error("Invalid tensor shape");
            }
            cv::Mat final_frame;
            if (model_type == "depth") {
                // Depth estimation: output raw depth values (CV_32FC1)
                int height = static_cast<int>(tensor_shape[2]);
                int width = static_cast<int>(tensor_shape[3]);
              
                // Create depth map from tensor data (single channel float)
                cv::Mat depth_map(height, width, CV_32FC1, const_cast<float*>(tensor_data));
              
                // Resize depth map to original image size (use LINEAR for depth)
                cv::Mat resized_depth;
                cv::resize(depth_map, resized_depth, frame.size(), 0, 0, cv::INTER_LINEAR);

                //// Only send out the depth
                final_frame = resized_depth;
                //// Debug: Show the blended result directly
                // std::unique_ptr<autoware_pov::common::DepthVisualizationEngine> viz_engine_ = 
                //     std::make_unique<autoware_pov::common::DepthVisualizationEngine>();
                // final_frame = viz_engine_->visualize(resized_depth);
            } else if (model_type == "segmentation") {
                cv::Mat mask;
 
#ifdef CUDA_FOUND
                // Try CUDA acceleration first
                bool cuda_success = autoware_pov::common::MasksVisualizationKernels::createMaskFromTensorCUDA(
                  tensor_data, tensor_shape, mask
                );
                if (!cuda_success)
#endif
                {
                    // CPU fallback: create mask from tensor
                    int height = static_cast<int>(tensor_shape[2]);
                    int width = static_cast<int>(tensor_shape[3]);
                    int channels = static_cast<int>(tensor_shape[1]);
      
                    mask = cv::Mat(height, width, CV_8UC1);
      
                    if (channels > 1) {
                        // Multi-class segmentation: argmax across channels (NCHW format)
                        for (int h = 0; h < height; ++h) {
                            for (int w = 0; w < width; ++w) {
                                float max_score = -1e9f;
                                uint8_t best_class = 0;
                                for (int c = 0; c < channels; ++c) {
                                    // NCHW format: tensor_data[batch=0][channel=c][height=h][width=w]
                                    float score = tensor_data[c * height * width + h * width + w];
                                    if (score > max_score) {
                                        max_score = score;
                                        best_class = static_cast<uint8_t>(c);
                                    }
                                }
                                // Convert class IDs for scene segmentation: Class 1 -> 255, others -> 0
                                mask.at<uint8_t>(h, w) = (best_class == 1) ? 255 : 0;
                            }
                        }
                    } else {
                        // Single channel: threshold for binary segmentation
                        for (int h = 0; h < height; ++h) {
                            for (int w = 0; w < width; ++w) {
                                float value = tensor_data[h * width + w];
                                mask.at<uint8_t>(h, w) = (value > 0.0f) ? 255 : 0;
                            }
                        }
                    }
                }
    
                // Resize mask to original image size (use NEAREST for masks)
                cv::Mat resized_mask;
                cv::resize(mask, resized_mask, frame.size(), 0, 0, cv::INTER_NEAREST);

                //// Only send out the mask
                final_frame = resized_mask;
                //// Debug: Show the blended result directly
                //std::unique_ptr<autoware_pov::common::MasksVisualizationEngine> viz_engine_ = 
                //    std::make_unique<autoware_pov::common::MasksVisualizationEngine>("scene");
                //final_frame = viz_engine_->visualize(resized_mask, frame);
            }

            // Benchmark: Inference done
            timer.recordInferenceEnd();

            // Publish the processed frame via Zenoh
            z_publisher_put_options_t options;
            z_publisher_put_options_default(&options);
            // Create attachment with frame metadata
            z_owned_bytes_t attachment_out;
            int output_bytes_info[] = {final_frame.rows, final_frame.cols, final_frame.type()};
            z_bytes_copy_from_buf(&attachment_out, (const uint8_t*)output_bytes_info, sizeof(output_bytes_info));
            options.attachment = z_move(attachment_out);
            // Create payload with pixel data and publish
            unsigned char* pixelPtr = final_frame.data;
            size_t dataSize = final_frame.total() * final_frame.elemSize();
            z_owned_bytes_t payload_out;
            z_bytes_copy_from_buf(&payload_out, pixelPtr, dataSize);
            z_publisher_put(z_loan(pub), z_move(payload_out), &options);

            // Benchmark: Output done
            timer.recordOutputEnd();

            // Release slice
            z_drop(z_move(zslice));
            // Release sample
            z_drop(z_move(sample));
        }
        
        // Cleanup
        z_drop(z_move(pub));
        z_drop(z_move(handler));
        z_drop(z_move(sub));
        z_drop(z_move(s));
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 