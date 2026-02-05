#include "onnxruntime_engine.hpp"
#include "onnxruntime_session.hpp"
#include "logging.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace autoware_pov::vision::autospeed
{

AutoSpeedOnnxEngine::AutoSpeedOnnxEngine(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  // Create session using factory (all provider logic is there)
  session_ = OnnxRuntimeSessionFactory::createSession(
    model_path, provider, precision, device_id, cache_dir
  );
  
  // Create memory info for CPU tensors
  memory_info_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
  );
  
  // Get input/output names and store them permanently
  Ort::AllocatorWithDefaultOptions allocator;
  
  // Input (typically "input" or "images")
  auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
  input_name_storage_ = std::string(input_name_allocated.get());
  input_names_.push_back(input_name_storage_.c_str());
  
  // Output (typically "output" or "output0")
  auto output_name_allocated = session_->GetOutputNameAllocated(0, allocator);
  output_name_storage_ = std::string(output_name_allocated.get());
  output_names_.push_back(output_name_storage_.c_str());
  
  // Get input shape
  auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_input_height_ = static_cast<int>(input_shape[2]);  // NCHW format
  model_input_width_ = static_cast<int>(input_shape[3]);
  
  // Get output shape (may be dynamic, will get actual shape after inference)
  auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  
  // Note: output_shape may be [-1, -1, -1] for dynamic shapes
  // We'll get actual dimensions after first inference
  LOG_INFO("[onnxrt_engine] Output shape (static/dynamic): [%lld, %lld, %lld]", 
           output_shape[0], output_shape[1], output_shape[2]);
  
  // Allocate input buffer only
  size_t input_size = model_input_width_ * model_input_height_ * 3;
  input_buffer_.resize(input_size);
  
  // Initialize output dimensions (will be set after first inference for dynamic shapes)
  model_output_channels_ = 0;
  model_output_predictions_ = 0;
  
  LOG_INFO("[onnxrt_engine] Engine initialized successfully");
  LOG_INFO("[onnxrt_engine] - Input: %dx%d", model_input_width_, model_input_height_);
}

AutoSpeedOnnxEngine::~AutoSpeedOnnxEngine()
{
  // Smart pointers handle cleanup automatically
}

void AutoSpeedOnnxEngine::preprocessAutoSpeed(const cv::Mat& input_image, float* buffer)
{
  // Store original dimensions for coordinate transformation
  orig_width_ = input_image.cols;
  orig_height_ = input_image.rows;

  // Step 1: Letterbox resize to 640x640 (maintain aspect ratio with padding)
  int target_w = model_input_width_;
  int target_h = model_input_height_;
  
  scale_ = std::min(
    static_cast<float>(target_w) / orig_width_,
    static_cast<float>(target_h) / orig_height_
  );
  
  int new_w = static_cast<int>(orig_width_ * scale_);
  int new_h = static_cast<int>(orig_height_ * scale_);
  
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  
  // Step 2: Create padded image with gray color (114, 114, 114)
  cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
  
  pad_x_ = (target_w - new_w) / 2;
  pad_y_ = (target_h - new_h) / 2;
  
  resized.copyTo(padded(cv::Rect(pad_x_, pad_y_, new_w, new_h)));
  
  // Step 3: Convert to float and normalize to [0, 1]
  cv::Mat float_image;
  padded.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
  
  // Step 4: Convert BGR to RGB and HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  // BGR → RGB: Reverse channel order
  int channel_size = target_h * target_w;
  memcpy(buffer, channels[2].data, channel_size * sizeof(float));  // R
  memcpy(buffer + channel_size, channels[1].data, channel_size * sizeof(float));  // G
  memcpy(buffer + 2 * channel_size, channels[0].data, channel_size * sizeof(float));  // B
}

bool AutoSpeedOnnxEngine::doInference(const cv::Mat& input_image)
{
  // Preprocess image
  preprocessAutoSpeed(input_image, input_buffer_.data());
  
  // Create input tensor
  std::vector<int64_t> input_shape = {1, 3, model_input_height_, model_input_width_};
  auto input_tensor = Ort::Value::CreateTensor<float>(
    *memory_info_,
    input_buffer_.data(),
    input_buffer_.size(),
    input_shape.data(),
    input_shape.size()
  );
  
  // Run inference (ONNX Runtime allocates output automatically)
  try {
    output_tensors_ = session_->Run(
      Ort::RunOptions{nullptr},
      input_names_.data(),
      &input_tensor,
      1,
      output_names_.data(),
      1
    );
  } catch (const Ort::Exception& e) {
    LOG_ERROR("[onnxrt_engine] Inference failed: %s", e.what());
    return false;
  }
  
  // Get actual output shape after inference (for dynamic shapes)
  if (!output_tensors_.empty()) {
    auto shape = output_tensors_[0].GetTensorTypeAndShapeInfo().GetShape();
    model_output_channels_ = static_cast<int>(shape[1]);
    model_output_predictions_ = static_cast<int>(shape[2]);
  }
  
  return true;
}

std::vector<Detection> AutoSpeedOnnxEngine::inference(
  const cv::Mat& input_image,
  float conf_thresh,
  float iou_thresh)
{
  // Run inference
  if (!doInference(input_image)) {
    LOG_ERROR("[onnxrt_engine] Inference failed");
    return {};
  }
  
  // Post-process and return detections
  return postProcess(conf_thresh, iou_thresh);
}

std::vector<Detection> AutoSpeedOnnxEngine::postProcess(float conf_thresh, float iou_thresh)
{
  std::vector<Detection> detections;
  
  if (output_tensors_.empty()) {
    LOG_ERROR("[onnxrt_engine] No output tensors available");
    return detections;
  }
  
  const float* raw_output = output_tensors_[0].GetTensorData<float>();
  int num_attrs = model_output_channels_;  // e.g., 8 (4 bbox + 4 classes)
  int num_boxes = model_output_predictions_;  // e.g., 8400

  // Parse detections
  for (int i = 0; i < num_boxes; ++i) {
    // Get class scores (skip first 4 for bbox)
    float max_score = 0.0f;
    int max_class = -1;
    for (int c = 4; c < num_attrs; ++c) {
      float score = raw_output[c * num_boxes + i];
      if (score > max_score) {
        max_score = score;
        max_class = c - 4;
      }
    }

    if (max_score < conf_thresh) continue;

    // Get bbox (center_x, center_y, width, height in letterbox space 0-640)
    float cx = raw_output[0 * num_boxes + i];
    float cy = raw_output[1 * num_boxes + i];
    float w = raw_output[2 * num_boxes + i];
    float h = raw_output[3 * num_boxes + i];

    // Convert xywh to xyxy (still in letterbox space)
    float x1_letterbox = cx - w / 2.0f;
    float y1_letterbox = cy - h / 2.0f;
    float x2_letterbox = cx + w / 2.0f;
    float y2_letterbox = cy + h / 2.0f;

    // Transform from letterbox space to original image space
    float x1_orig = (x1_letterbox - pad_x_) / scale_;
    float y1_orig = (y1_letterbox - pad_y_) / scale_;
    float x2_orig = (x2_letterbox - pad_x_) / scale_;
    float y2_orig = (y2_letterbox - pad_y_) / scale_;

    // Clamp to image bounds
    x1_orig = std::max(0.0f, std::min(static_cast<float>(orig_width_), x1_orig));
    y1_orig = std::max(0.0f, std::min(static_cast<float>(orig_height_), y1_orig));
    x2_orig = std::max(0.0f, std::min(static_cast<float>(orig_width_), x2_orig));
    y2_orig = std::max(0.0f, std::min(static_cast<float>(orig_height_), y2_orig));

    Detection det;
    det.x1 = x1_orig;
    det.y1 = y1_orig;
    det.x2 = x2_orig;
    det.y2 = y2_orig;
    det.confidence = max_score;
    det.class_id = max_class;

    detections.push_back(det);
  }

  // Apply NMS
  detections = applyNMS(detections, iou_thresh);

  return detections;
}

float AutoSpeedOnnxEngine::computeIoU(const Detection& a, const Detection& b)
{
  float inter_x1 = std::max(a.x1, b.x1);
  float inter_y1 = std::max(a.y1, b.y1);
  float inter_x2 = std::min(a.x2, b.x2);
  float inter_y2 = std::min(a.y2, b.y2);
  
  float inter_width = std::max(0.0f, inter_x2 - inter_x1);
  float inter_height = std::max(0.0f, inter_y2 - inter_y1);
  float inter_area = inter_width * inter_height;
  
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - inter_area;
  
  return union_area > 0 ? inter_area / union_area : 0.0f;
}

std::vector<Detection> AutoSpeedOnnxEngine::applyNMS(
  std::vector<Detection>& detections,
  float iou_thresh)
{
  // Sort by confidence (descending)
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  
  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);
  
  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;
    
    result.push_back(detections[i]);
    
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j]) continue;
      
      // Only suppress same class
      if (detections[i].class_id == detections[j].class_id) {
        if (computeIoU(detections[i], detections[j]) > iou_thresh) {
          suppressed[j] = true;
        }
      }
    }
  }
  
  return result;
}

const float* AutoSpeedOnnxEngine::getRawTensorData() const
{
  if (output_tensors_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call inference() first.");
  }
  return output_tensors_[0].GetTensorData<float>();
}

std::vector<int64_t> AutoSpeedOnnxEngine::getTensorShape() const
{
  if (output_tensors_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call inference() first.");
  }
  return output_tensors_[0].GetTensorTypeAndShapeInfo().GetShape();
}

}  // namespace autoware_pov::vision::autospeed

