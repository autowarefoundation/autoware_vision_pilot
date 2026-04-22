#ifndef AUTOWARE_POV_VISION_EGOLANES_TENSORRT_AUTOSTEER_ENGINE_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_TENSORRT_AUTOSTEER_ENGINE_HPP_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <memory>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief TensorRT Logger for AutoSteer
 */
class AutoSteerTensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override;
};

/**
 * @brief AutoSteer TensorRT Inference Engine
 * 
 * Direct TensorRT implementation (no ONNX Runtime dependency).
 * Takes concatenated EgoLanes raw outputs from two consecutive frames
 * and predicts steering angle.
 * 
 * Input: [1, 6, 80, 160] - concatenated (t-1, t) EgoLanes outputs
 * Output: Steering angle (degrees)
 * 
 * Supports FP16 and FP32 precision.
 */
class AutoSteerTensorRTEngine
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
  AutoSteerTensorRTEngine(
    const std::string& model_path,
    const std::string& precision = "fp16",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  ~AutoSteerTensorRTEngine();

  /**
   * @brief Run AutoSteer inference
   * 
   * @param concat_input Concatenated tensor [1, 6, 80, 160] 
   *                     = [t-1 EgoLanes output, t EgoLanes output]
   * @return Steering angle in degrees (-30 to +30)
   */
  float inference(const std::vector<float>& concat_input);

  /**
   * @brief Get model input dimensions
   */
  int getInputChannels() const { return model_input_channels_; }
  int getInputHeight() const { return model_input_height_; }
  int getInputWidth() const { return model_input_width_; }
  
  /**
   * @brief Get raw output logits for debugging
   * @return Vector of 61 logit values (one per steering class)
   */
  std::vector<float> getRawOutputLogits() const;

private:
  /**
   * @brief Run TensorRT inference
   */
  bool doInference(const std::vector<float>& input_buffer);

  /**
   * @brief Post-process raw output to steering angle
   * 
   * Applies argmax and converts to degrees: argmax(logits) - 30
   */
  float postProcess();

  /**
   * @brief Build TensorRT engine from ONNX model
   */
  void buildEngineFromOnnx(const std::string& onnx_path, const std::string& precision);

  /**
   * @brief Load pre-built TensorRT engine
   */
  void loadEngine(const std::string& engine_path);

  // TensorRT components
  AutoSteerTensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // CUDA resources
  cudaStream_t stream_;
  void* input_buffer_gpu_;
  void* output_buffer_gpu_;  // Buffer for the output we care about (second output)
  void* output_buffer_gpu_other_;  // Buffer for the other output (first output, unused)
  std::vector<float> output_buffer_host_;

  // Model dimensions
  int model_input_channels_;  // 6 (two 3-channel EgoLanes outputs)
  int model_input_height_;    // 80
  int model_input_width_;      // 160
  int model_output_classes_;  // 61 (steering classes: -30 to +30)

  // Engine cache path
  std::string cache_dir_;
  std::string engine_cache_path_;

  // Output tensor name (use second output for current frame prediction)
  std::string output_tensor_name_;
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_TENSORRT_AUTOSTEER_ENGINE_HPP_

