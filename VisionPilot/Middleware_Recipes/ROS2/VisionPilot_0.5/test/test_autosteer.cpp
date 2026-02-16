/**
 * @file test_autosteer.cpp
 * @brief Minimal test script for AutoSteer inference with visualization
 * 
 * Tests AutoSteer model on video input and displays steering angle predictions
 */

#include "../include/inference/onnxruntime_engine.hpp"
#include "../include/inference/autosteer_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace autoware_pov::vision::egolanes;

// Debug helper functions
void printTensorStats(const std::vector<float>& tensor, const std::string& name, int frame_num) {
    if (tensor.empty()) {
        std::cout << "[DEBUG Frame " << frame_num << "] " << name << ": EMPTY" << std::endl;
        return;
    }
    
    float min_val = *std::min_element(tensor.begin(), tensor.end());
    float max_val = *std::max_element(tensor.begin(), tensor.end());
    float sum = std::accumulate(tensor.begin(), tensor.end(), 0.0f);
    float mean = sum / tensor.size();
    
    float variance = 0.0f;
    for (float val : tensor) {
        variance += (val - mean) * (val - mean);
    }
    float std_dev = std::sqrt(variance / tensor.size());
    
    std::cout << "[DEBUG Frame " << frame_num << "] " << name << " Stats:" << std::endl;
    std::cout << "  Size: " << tensor.size() << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
    std::cout << "  Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
    std::cout << "  Std: " << std::fixed << std::setprecision(6) << std_dev << std::endl;
    std::cout << "  First 10 values: ";
    for (size_t i = 0; i < std::min(10UL, tensor.size()); ++i) {
        std::cout << std::fixed << std::setprecision(4) << tensor[i] << " ";
    }
    std::cout << std::endl;
}

void compareTensors(const std::vector<float>& t1, const std::vector<float>& t2, const std::string& name) {
    if (t1.size() != t2.size()) {
        std::cout << "[DEBUG] " << name << " comparison: SIZES DIFFER (" 
                  << t1.size() << " vs " << t2.size() << ")" << std::endl;
        return;
    }
    
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_different = 0;
    
    for (size_t i = 0; i < t1.size(); ++i) {
        float diff = std::abs(t1[i] - t2[i]);
        if (diff > 1e-6f) {
            num_different++;
        }
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    
    float mean_diff = sum_diff / t1.size();
    
    std::cout << "[DEBUG] " << name << " Comparison:" << std::endl;
    std::cout << "  Max difference: " << std::fixed << std::setprecision(6) << max_diff << std::endl;
    std::cout << "  Mean difference: " << std::fixed << std::setprecision(6) << mean_diff << std::endl;
    std::cout << "  Different elements: " << num_different << " / " << t1.size() 
              << " (" << (100.0f * num_different / t1.size()) << "%)" << std::endl;
    
    if (num_different == 0) {
        std::cout << "  WARNING: Tensors are IDENTICAL!" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <egolanes_model> <autosteer_model> [cache_dir]\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " video.mp4 egolanes.onnx autosteer.onnx ./trt_cache\n";
        return 1;
    }

    std::string video_path = argv[1];
    std::string egolanes_model = argv[2];
    std::string autosteer_model = argv[3];
    std::string cache_dir = (argc >= 5) ? argv[4] : "./trt_cache";
    
    // Default: TensorRT FP16, but allow override
    std::string provider = (argc >= 6) ? argv[5] : "tensorrt";
    std::string precision = "fp16";
    int device_id = 0;

    std::cout << "========================================\n";
    std::cout << "AutoSteer Test Script\n";
    std::cout << "========================================\n";
    std::cout << "Video: " << video_path << "\n";
    std::cout << "EgoLanes Model: " << egolanes_model << "\n";
    std::cout << "AutoSteer Model: " << autosteer_model << "\n";
    std::cout << "Provider: " << provider << " | Precision: " << precision << "\n";
    std::cout << "Cache: " << cache_dir << "\n";
    std::cout << "========================================\n\n";

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << video_path << std::endl;
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << frame_width << "x" << frame_height 
              << " @ " << fps << " fps, " << total_frames << " frames\n\n";

    // Initialize EgoLanes engine
    std::cout << "Loading EgoLanes model..." << std::endl;
    EgoLanesOnnxEngine egolanes_engine(egolanes_model, provider, precision, device_id, cache_dir);
    std::cout << "EgoLanes initialized!\n" << std::endl;

    // Initialize AutoSteer engine
    std::cout << "Loading AutoSteer model..." << std::endl;
    AutoSteerOnnxEngine autosteer_engine(autosteer_model, provider, precision, device_id, cache_dir);
    std::cout << "AutoSteer initialized!\n" << std::endl;

    // Temporal buffer for AutoSteer
    const int EGOLANES_TENSOR_SIZE = 3 * 80 * 160;  // 38,400 floats
    std::vector<std::vector<float>> tensor_buffer;
    tensor_buffer.reserve(2);
    std::vector<float> autosteer_input_buffer(6 * 80 * 160);  // Pre-allocated

    // Debug: Verify tensor size
    std::cout << "[DEBUG] Expected EgoLanes tensor size: " << EGOLANES_TENSOR_SIZE << " floats" << std::endl;
    std::cout << "[DEBUG] Expected AutoSteer input size: " << (6 * 80 * 160) << " floats" << std::endl;
    
    // Get actual tensor shape from EgoLanes (also serves as warm-up)
    cv::Mat dummy_frame(720, 1280, CV_8UC3, cv::Scalar(128, 128, 128));
    if (dummy_frame.rows > 420) {
        dummy_frame = dummy_frame(cv::Rect(0, 420, dummy_frame.cols, dummy_frame.rows - 420));
    }
    egolanes_engine.inference(dummy_frame, 0.0f);
    
    // Warm-up AutoSteer
    if (provider == "tensorrt") {
        std::cout << "Warming up AutoSteer engine (building TensorRT engine)..." << std::endl;
        std::vector<float> dummy_input(6 * 80 * 160, 0.5f);
        autosteer_engine.inference(dummy_input);
        std::cout << "Warm-up complete!\n" << std::endl;
    }
    auto tensor_shape = egolanes_engine.getTensorShape();
    std::cout << "[DEBUG] Actual EgoLanes output shape: [";
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
        std::cout << tensor_shape[i];
        if (i < tensor_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    int actual_tensor_size = 1;
    for (size_t i = 1; i < tensor_shape.size(); ++i) {
        actual_tensor_size *= static_cast<int>(tensor_shape[i]);
    }
    std::cout << "[DEBUG] Actual tensor size (excluding batch): " << actual_tensor_size << " floats" << std::endl;
    
    if (actual_tensor_size != EGOLANES_TENSOR_SIZE) {
        std::cerr << "[ERROR] Tensor size mismatch! Expected " << EGOLANES_TENSOR_SIZE 
                  << ", got " << actual_tensor_size << std::endl;
        std::cerr << "[ERROR] This will cause incorrect data copying!" << std::endl;
    }
    std::cout << std::endl;

    // OpenCV window
    cv::namedWindow("AutoSteer Test", cv::WINDOW_NORMAL);
    cv::resizeWindow("AutoSteer Test", 1280, 720);

    int frame_number = 0;
    cv::Mat frame;
    std::vector<float> prev_tensor;  // For comparison

    std::cout << "Processing video... (Press 'q' to quit, 's' to step frame-by-frame)" << std::endl;
    std::cout << "Debug mode: ON (will print detailed statistics)\n" << std::endl;

    while (cap.read(frame) && !frame.empty()) {
        // Crop top 420 pixels (same as main pipeline)
        if (frame.rows > 420) {
            frame = frame(cv::Rect(0, 420, frame.cols, frame.rows - 420));
        }

        // Run EgoLanes inference
        LaneSegmentation lanes = egolanes_engine.inference(frame, 0.0f);

        // Get raw tensor for AutoSteer
        const float* raw_tensor = egolanes_engine.getRawTensorData();
        if (raw_tensor == nullptr) {
            std::cerr << "[ERROR Frame " << frame_number << "] Raw tensor pointer is NULL!" << std::endl;
            continue;
        }
        
        // Verify tensor shape
        auto tensor_shape = egolanes_engine.getTensorShape();
        int actual_size = 1;
        for (size_t i = 1; i < tensor_shape.size(); ++i) {
            actual_size *= static_cast<int>(tensor_shape[i]);
        }
        
        // Use actual size instead of hardcoded size
        int copy_size = std::min(EGOLANES_TENSOR_SIZE, actual_size);
        std::vector<float> current_tensor(copy_size);
        std::memcpy(current_tensor.data(), raw_tensor, copy_size * sizeof(float));
        
        // Debug: Print tensor statistics
        if (frame_number < 3 || frame_number % 30 == 0) {
            printTensorStats(current_tensor, "EgoLanes Raw Tensor", frame_number);
        }
        
        // Debug: Compare with previous frame
        if (!prev_tensor.empty() && prev_tensor.size() == current_tensor.size()) {
            if (frame_number < 3 || frame_number % 30 == 0) {
                compareTensors(prev_tensor, current_tensor, "EgoLanes Frame-to-Frame");
            }
        }
        prev_tensor = current_tensor;

        // Store in buffer
        tensor_buffer.push_back(current_tensor);

        // Run AutoSteer when we have 2 frames
        float steering_angle = 0.0f;
        bool autosteer_valid = false;

        if (tensor_buffer.size() >= 2) {
            // Keep only last 2 frames
            if (tensor_buffer.size() > 2) {
                tensor_buffer.erase(tensor_buffer.begin());
            }

            // Debug: Check buffer sizes
            if (frame_number < 3 || frame_number % 30 == 0) {
                std::cout << "[DEBUG Frame " << frame_number << "] Buffer status:" << std::endl;
                std::cout << "  Buffer size: " << tensor_buffer.size() << std::endl;
                std::cout << "  t-1 tensor size: " << tensor_buffer[0].size() << std::endl;
                std::cout << "  t tensor size: " << tensor_buffer[1].size() << std::endl;
                printTensorStats(tensor_buffer[0], "Buffer t-1", frame_number);
                printTensorStats(tensor_buffer[1], "Buffer t", frame_number);
            }

            // Concatenate t-1 and t
            int t1_size = static_cast<int>(tensor_buffer[0].size());
            int t_size = static_cast<int>(tensor_buffer[1].size());
            int total_size = t1_size + t_size;
            
            if (total_size > static_cast<int>(autosteer_input_buffer.size())) {
                std::cerr << "[ERROR Frame " << frame_number << "] Buffer overflow! "
                          << "Need " << total_size << " floats, have " 
                          << autosteer_input_buffer.size() << std::endl;
                continue;
            }
            
            std::memcpy(autosteer_input_buffer.data(), 
                       tensor_buffer[0].data(),  // t-1
                       t1_size * sizeof(float));
            
            std::memcpy(autosteer_input_buffer.data() + t1_size,
                       tensor_buffer[1].data(),  // t
                       t_size * sizeof(float));
            
            // Debug: Print concatenated input statistics
            std::vector<float> autosteer_input_vec(autosteer_input_buffer.begin(), 
                                                   autosteer_input_buffer.begin() + total_size);
            if (frame_number < 3 || frame_number % 30 == 0) {
                printTensorStats(autosteer_input_vec, "AutoSteer Input (concatenated)", frame_number);
            }

            // Run AutoSteer
            std::vector<float> autosteer_input_for_inference(autosteer_input_buffer.begin(), 
                                                             autosteer_input_buffer.begin() + total_size);
            steering_angle = autosteer_engine.inference(autosteer_input_for_inference);
            autosteer_valid = true;
            
            // Debug: ALWAYS print logits for first few frames to diagnose the issue
            if (frame_number < 5 || frame_number % 30 == 0) {
                std::vector<float> logits = autosteer_engine.getRawOutputLogits();
                if (!logits.empty()) {
                    std::cout << "[DEBUG Frame " << frame_number << "] AutoSteer Output Logits:" << std::endl;
                    std::cout << "  Size: " << logits.size() << " (expected 61)" << std::endl;
                    
                    float min_logit = *std::min_element(logits.begin(), logits.end());
                    float max_logit = *std::max_element(logits.begin(), logits.end());
                    float sum_logit = std::accumulate(logits.begin(), logits.end(), 0.0f);
                    float mean_logit = sum_logit / logits.size();
                    
                    std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_logit << std::endl;
                    std::cout << "  Max: " << std::fixed << std::setprecision(6) << max_logit << std::endl;
                    std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean_logit << std::endl;
                    std::cout << "  ALL 61 logits: ";
                    for (size_t i = 0; i < logits.size(); ++i) {
                        std::cout << std::fixed << std::setprecision(3) << logits[i];
                        if (i < logits.size() - 1) std::cout << " ";
                    }
                    std::cout << std::endl;
                    
                    // Find argmax
                    int argmax_idx = 0;
                    float argmax_val = logits[0];
                    for (size_t i = 1; i < logits.size(); ++i) {
                        if (logits[i] > argmax_val) {
                            argmax_val = logits[i];
                            argmax_idx = static_cast<int>(i);
                        }
                    }
                    
                    std::cout << "  Argmax index: " << argmax_idx << " (steering = " 
                              << (argmax_idx - 30) << " deg)" << std::endl;
                    std::cout << "  Argmax value: " << std::fixed << std::setprecision(6) 
                              << argmax_val << std::endl;
                    
                    // Show top 5 classes by probability
                    std::vector<std::pair<float, int>> logit_pairs;
                    for (size_t i = 0; i < logits.size(); ++i) {
                        logit_pairs.push_back({logits[i], static_cast<int>(i)});
                    }
                    std::sort(logit_pairs.rbegin(), logit_pairs.rend());  // Sort descending
                    
                    std::cout << "  Top 5 classes:" << std::endl;
                    for (int i = 0; i < 5 && i < static_cast<int>(logit_pairs.size()); ++i) {
                        int class_idx = logit_pairs[i].second;
                        float value = logit_pairs[i].first;
                        std::cout << "    Class " << class_idx << " (steering=" << (class_idx - 30) 
                                  << " deg): " << std::fixed << std::setprecision(6) << value << std::endl;
                    }
                    
                    // Show first 10 and last 10 logits
                    std::cout << "  First 10 logits: ";
                    for (size_t i = 0; i < std::min(10UL, logits.size()); ++i) {
                        std::cout << std::fixed << std::setprecision(3) << logits[i] << " ";
                    }
                    std::cout << std::endl;
                    
                    std::cout << "  Last 10 logits: ";
                    for (size_t i = std::max(0UL, logits.size() - 10); i < logits.size(); ++i) {
                        std::cout << std::fixed << std::setprecision(3) << logits[i] << " ";
                    }
                    std::cout << std::endl;
                    
                    // Check if all logits are the same (would indicate a problem)
                    bool all_same = true;
                    for (size_t i = 1; i < logits.size(); ++i) {
                        if (std::abs(logits[i] - logits[0]) > 1e-6f) {
                            all_same = false;
                            break;
                        }
                    }
                    if (all_same) {
                        std::cout << "  [WARNING] All logits are IDENTICAL! Model not responding to input." << std::endl;
                    }
                }
            }

            std::cout << "[Frame " << frame_number << "] "
                      << "Steering: " << std::fixed << std::setprecision(2) 
                      << steering_angle << " deg" << std::endl;
            
            // Debug: Track steering angles
            static std::vector<float> steering_history;
            steering_history.push_back(steering_angle);
            if (steering_history.size() > 10) {
                steering_history.erase(steering_history.begin());
            }
            
            if (frame_number < 3 || frame_number % 30 == 0) {
                std::cout << "[DEBUG Frame " << frame_number << "] Recent steering angles: ";
                for (float s : steering_history) {
                    std::cout << std::fixed << std::setprecision(2) << s << " ";
                }
                std::cout << std::endl;
                
                // Check if steering is constant
                if (steering_history.size() >= 3) {
                    bool all_same = true;
                    for (size_t i = 1; i < steering_history.size(); ++i) {
                        if (std::abs(steering_history[i] - steering_history[0]) > 0.01f) {
                            all_same = false;
                            break;
                        }
                    }
                    if (all_same) {
                        std::cout << "[WARNING Frame " << frame_number 
                                  << "] Steering angle is CONSTANT! All recent values are: " 
                                  << steering_history[0] << std::endl;
                    }
                }
            }
        } else {
            std::cout << "[Frame " << frame_number << "] "
                      << "Skipped (waiting for temporal buffer, size=" 
                      << tensor_buffer.size() << ")" << std::endl;
        }

        // Visualize
        cv::Mat vis_frame = frame.clone();

        // Draw steering angle text
        std::stringstream ss;
        ss << "Frame: " << frame_number;
        if (autosteer_valid) {
            ss << " | Steering: " << std::fixed << std::setprecision(2) << steering_angle << " deg";
        } else {
            ss << " | Waiting for buffer...";
        }

        // Draw text with background for visibility
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::rectangle(vis_frame, 
                     cv::Point(10, 10), 
                     cv::Point(20 + text_size.width, 50 + text_size.height),
                     cv::Scalar(0, 0, 0), -1);
        cv::putText(vis_frame, ss.str(), cv::Point(15, 40),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Draw steering angle indicator (simple bar)
        if (autosteer_valid) {
            int center_x = vis_frame.cols / 2;
            int bar_width = 400;
            int bar_height = 30;
            int bar_y = vis_frame.rows - 60;

            // Background bar
            cv::rectangle(vis_frame,
                         cv::Point(center_x - bar_width/2, bar_y),
                         cv::Point(center_x + bar_width/2, bar_y + bar_height),
                         cv::Scalar(100, 100, 100), -1);

            // Steering indicator (-30 to +30 degrees mapped to bar width)
            float normalized = steering_angle / 30.0f;  // -1.0 to +1.0
            normalized = std::max(-1.0f, std::min(1.0f, normalized));  // Clamp
            int indicator_x = center_x + static_cast<int>(normalized * bar_width / 2);

            // Color: green for center, red for extremes
            cv::Scalar color = (std::abs(normalized) < 0.3) ? 
                              cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::circle(vis_frame, cv::Point(indicator_x, bar_y + bar_height/2), 15, color, -1);
            cv::line(vis_frame, 
                    cv::Point(center_x, bar_y - 5),
                    cv::Point(center_x, bar_y + bar_height + 5),
                    cv::Scalar(255, 255, 255), 2);
        }

        cv::imshow("AutoSteer Test", vis_frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            std::cout << "\nQuitting..." << std::endl;
            break;
        } else if (key == 's') {
            // Step mode: wait for key press
            cv::waitKey(0);
        }

        frame_number++;
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nProcessed " << frame_number << " frames." << std::endl;
    return 0;
}

