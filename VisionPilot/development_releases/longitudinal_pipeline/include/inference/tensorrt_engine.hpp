#ifndef AUTOWARE_POV_VISION_EGOLANES_TENSORRT_ENGINE_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_TENSORRT_ENGINE_HPP_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "inference/lane_segmentation.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace autoware_pov::vision::egolanes
{

// Forward declaration
struct LaneSegmentation;

/**
 * @brief TensorRT Logger for EgoLanes
 */
class TensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override;
};

/**
 * @brief EgoLanes TensorRT Inference Engine
 * 
 * Direct TensorRT implementation (no ONNX Runtime dependency).
 * Supports FP16 and FP32 precision.
 * 
 * Handles complete inference pipeline:
 * 1. Preprocessing (resize to 320x640, normalize)
 * 2. Model inference via TensorRT
 * 3. Post-processing (thresholding, channel extraction)
 */
class EgoLanesTensorRTEngine
{
public:
  /**
   * @brief Constructor
   * 
   * @param model_path Path to ONNX model (.onnx file)
   * @param precision Precision mode: "fp32" or "fp16"
   * @param device_id GPU device ID (default: 0)
   * @param cache_dir TensorRT engine cache directory (default: ./trt_cache)
   */
  EgoLanesTensorRTEngine(
    const std::string& model_path,
    const std::string& precision = "fp16",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  ~EgoLanesTensorRTEngine();

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
   * @brief Run TensorRT inference
   */
  bool doInference(const cv::Mat& input_image);

  /**
   * @brief Post-process raw output
   * 
   * Applies threshold and extracts lane masks from 3-channel output
   */
  LaneSegmentation postProcess(float threshold);

  /**
   * @brief Build TensorRT engine from ONNX model
   */
  void buildEngineFromOnnx(const std::string& onnx_path, const std::string& precision);

  /**
   * @brief Load pre-built TensorRT engine
   */
  void loadEngine(const std::string& engine_path);

  // TensorRT components
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // CUDA resources
  cudaStream_t stream_;
  void* input_buffer_gpu_;
  void* output_buffer_gpu_;
  std::vector<float> output_buffer_host_;

  // Model dimensions
  int model_input_width_;    // 640
  int model_input_height_;   // 320
  int model_output_width_;   // 640
  int model_output_height_;  // 320
  int model_output_channels_; // 3 (ego_left, ego_right, other_lanes)
  
  // Engine cache path
  std::string cache_dir_;
  std::string engine_cache_path_;

  // ImageNet normalization constants
  static constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
  static constexpr std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_TENSORRT_ENGINE_HPP_

