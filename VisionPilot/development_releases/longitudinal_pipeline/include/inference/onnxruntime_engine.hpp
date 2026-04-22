#ifndef AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_ENGINE_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_ENGINE_HPP_

#include <onnxruntime_cxx_api.h>
#include "inference/lane_segmentation.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief EgoLanes ONNX Runtime Inference Engine
 * 
 * Supports multiple execution providers:
 * - CPU: Default CPU execution
 * - TensorRT: GPU-accelerated with FP16/FP32
 * 
 * Handles complete inference pipeline:
 * 1. Preprocessing (resize to 320x640, normalize)
 * 2. Model inference via ONNX Runtime
 * 3. Post-processing (thresholding, channel extraction)
 */
class EgoLanesOnnxEngine
{
public:
  /**
   * @brief Constructor
   * 
   * @param model_path Path to ONNX model (.onnx file)
   * @param provider Execution provider: "cpu" or "tensorrt"
   * @param precision Precision mode: "fp32" or "fp16" (TensorRT only)
   * @param device_id GPU device ID (TensorRT only, default: 0)
   * @param cache_dir TensorRT engine cache directory (default: ./trt_cache)
   */
  EgoLanesOnnxEngine(
    const std::string& model_path,
    const std::string& provider = "cpu",
    const std::string& precision = "fp32",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  ~EgoLanesOnnxEngine();

  /**
   * @brief Run complete inference pipeline
   * 
   * @param input_image Input image (BGR, any resolution)
   * @param threshold Segmentation threshold (default: 0.0)
   * @return Lane segmentation masks in model resolution (320x640)
   */
  LaneSegmentation inference(
    const cv::Mat& input_image,
    float threshold = 0.0f
  );

  /**
   * @brief Get raw tensor output (for advanced users)
   * @return Pointer to raw output tensor [1, 3, height, width]
   */
  const float* getRawTensorData() const;

  /**
   * @brief Get output tensor shape
   * @return Shape as [batch, channels, height, width]
   */
  std::vector<int64_t> getTensorShape() const;

  // Model input dimensions (320x640 for EgoLanes)
  int getInputWidth() const { return model_input_width_; }
  int getInputHeight() const { return model_input_height_; }

  // Model output dimensions
  int getOutputWidth() const { return model_output_width_; }
  int getOutputHeight() const { return model_output_height_; }

private:
  /**
   * @brief Preprocess image for EgoLanes
   * 
   * Resizes to 320x640, converts to RGB, normalizes with ImageNet stats, converts to CHW
   */
  void preprocessEgoLanes(const cv::Mat& input_image, float* buffer);

  /**
   * @brief Run ONNX Runtime inference
   */
  bool doInference(const cv::Mat& input_image);

  /**
   * @brief Post-process raw output
   * 
   * Applies threshold and extracts lane masks from 3-channel output
   */
  LaneSegmentation postProcess(float threshold);

  // ONNX Runtime components
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  
  // Input/Output tensor names (storage + pointers)
  std::string input_name_storage_;
  std::string output_name_storage_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  
  // Model dimensions
  int model_input_width_;    // 640
  int model_input_height_;   // 320
  int model_output_width_;   // 640
  int model_output_height_;  // 320
  int model_output_channels_; // 3 (ego_left, ego_right, other_lanes)
  
  // Buffers
  std::vector<float> input_buffer_;
  std::vector<Ort::Value> output_tensors_;  // Output managed by ONNX Runtime
  
  // ImageNet normalization constants
  static constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
  static constexpr std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_ENGINE_HPP_

