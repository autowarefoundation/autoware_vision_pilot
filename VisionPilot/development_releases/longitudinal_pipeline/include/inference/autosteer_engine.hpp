#ifndef AUTOWARE_POV_VISION_EGOLANES_AUTOSTEER_ENGINE_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_AUTOSTEER_ENGINE_HPP_

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief AutoSteer ONNX Runtime Inference Engine
 * 
 * Takes concatenated EgoLanes raw outputs from two consecutive frames
 * and predicts steering angle.
 * 
 * Input: [1, 6, 80, 160] - concatenated (t-1, t) EgoLanes outputs
 * Output: Steering angle (degrees)
 * 
 * Supports CPU and TensorRT execution providers.
 */
class AutoSteerOnnxEngine
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
  AutoSteerOnnxEngine(
    const std::string& model_path,
    const std::string& provider = "cpu",
    const std::string& precision = "fp32",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  ~AutoSteerOnnxEngine();

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
   * @brief Run ONNX Runtime inference
   */
  bool doInference(const std::vector<float>& input_buffer);

  /**
   * @brief Post-process raw output to steering angle
   * 
   * Applies argmax and converts to degrees: argmax(logits) - 30
   */
  float postProcess();

  // ONNX Runtime components
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  
  // Input/Output tensor names (storage + pointers)
  std::string input_name_storage_;
  std::string output_name_storage_;  // Kept for backward compatibility (points to second output)
  std::vector<std::string> output_names_storage_;  // Stores both output names
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  
  // Model dimensions
  int model_input_channels_;  // 6 (two 3-channel EgoLanes outputs)
  int model_input_height_;    // 80
  int model_input_width_;     // 160
  int model_output_classes_;  // 61 (steering classes: -30 to +30)
  
  // Output buffer
  std::vector<Ort::Value> output_tensors_;
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_AUTOSTEER_ENGINE_HPP_

