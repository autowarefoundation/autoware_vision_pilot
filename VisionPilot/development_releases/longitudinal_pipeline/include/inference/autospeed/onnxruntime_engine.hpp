#ifndef AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_ENGINE_HPP_
#define AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_ENGINE_HPP_

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace autoware_pov::vision::autospeed
{

struct Detection
{
  float x1, y1, x2, y2;  // Bounding box in original image coordinates
  float confidence;
  int class_id;
};

/**
 * @brief AutoSpeed ONNX Runtime Inference Engine
 * 
 * Supports multiple execution providers:
 * - CPU: Default CPU execution
 * - TensorRT: GPU-accelerated with FP16/FP32
 * 
 * Handles complete inference pipeline:
 * 1. Letterbox preprocessing (640x640)
 * 2. Model inference via ONNX Runtime
 * 3. Post-processing with NMS
 */
class AutoSpeedOnnxEngine
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
  AutoSpeedOnnxEngine(
    const std::string& model_path,
    const std::string& provider = "cpu",
    const std::string& precision = "fp32",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  ~AutoSpeedOnnxEngine();

  /**
   * @brief Run complete inference pipeline
   * 
   * @param input_image Input image (BGR, any resolution)
   * @param conf_thresh Confidence threshold (default: 0.25)
   * @param iou_thresh NMS IoU threshold (default: 0.45)
   * @return Vector of detections in original image coordinates
   */
  std::vector<Detection> inference(
    const cv::Mat& input_image,
    float conf_thresh = 0.25f,
    float iou_thresh = 0.45f
  );

  /**
   * @brief Get raw tensor output (for advanced users)
   * @return Pointer to raw output tensor [1, num_predictions, num_attributes]
   */
  const float* getRawTensorData() const;

  /**
   * @brief Get output tensor shape
   * @return Shape as [batch, num_predictions, num_attributes]
   */
  std::vector<int64_t> getTensorShape() const;

  // Model input dimensions (typically 640x640 for AutoSpeed)
  int getInputWidth() const { return model_input_width_; }
  int getInputHeight() const { return model_input_height_; }

private:
  /**
   * @brief Letterbox preprocessing for AutoSpeed
   * 
   * Resizes image to 640x640 with padding, normalizes to [0,1], converts to CHW RGB
   */
  void preprocessAutoSpeed(const cv::Mat& input_image, float* buffer);

  /**
   * @brief Run ONNX Runtime inference
   */
  bool doInference(const cv::Mat& input_image);

  /**
   * @brief Post-process raw output
   * 
   * Parses YOLO-format output [num_attrs, num_boxes] and applies NMS
   */
  std::vector<Detection> postProcess(float conf_thresh, float iou_thresh);

  /**
   * @brief Apply Non-Maximum Suppression
   */
  std::vector<Detection> applyNMS(std::vector<Detection>& detections, float iou_thresh);

  /**
   * @brief Compute IoU between two detections
   */
  float computeIoU(const Detection& a, const Detection& b);

  // ONNX Runtime components
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  
  // Input/Output tensor names (storage + pointers)
  std::string input_name_storage_;
  std::string output_name_storage_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  
  // Model dimensions
  int model_input_width_;
  int model_input_height_;
  int model_output_channels_;      // e.g., 8 (4 bbox + 4 classes)
  int model_output_predictions_;   // e.g., 8400
  
  // Buffers
  std::vector<float> input_buffer_;
  std::vector<Ort::Value> output_tensors_;  // Output managed by ONNX Runtime
  
  // Letterbox transform parameters (for coordinate mapping)
  float scale_;
  int pad_x_, pad_y_;
  int orig_width_, orig_height_;
};

}  // namespace autoware_pov::vision::autospeed

#endif  // AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_ENGINE_HPP_

