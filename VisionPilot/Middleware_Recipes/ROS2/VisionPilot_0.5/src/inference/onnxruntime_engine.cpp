#include "inference/onnxruntime_engine.hpp"
#include "inference/onnxruntime_session.hpp"
#include <stdexcept>
#include <cstdio>

// Simple logging macros (standalone version)
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

EgoLanesOnnxEngine::EgoLanesOnnxEngine(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  // Create session using factory (all provider logic is there)
  // Use 1GB workspace for EgoLanes
  session_ = OnnxRuntimeSessionFactory::createSession(
    model_path, provider, precision, device_id, cache_dir, 1.0, "egolanes_"
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
  model_input_height_ = static_cast<int>(input_shape[2]);  // NCHW format: [1, 3, 320, 640]
  model_input_width_ = static_cast<int>(input_shape[3]);
  
  // Get output shape
  auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_output_channels_ = static_cast<int>(output_shape[1]);  // [1, 3, 320, 640]
  model_output_height_ = static_cast<int>(output_shape[2]);
  model_output_width_ = static_cast<int>(output_shape[3]);
  
  LOG_INFO("[onnxrt_engine] Output shape: [1, %d, %d, %d]", 
           model_output_channels_, model_output_height_, model_output_width_);
  
  // Allocate input buffer
  size_t input_size = model_input_width_ * model_input_height_ * 3;
  input_buffer_.resize(input_size);
  
  LOG_INFO("[onnxrt_engine] EgoLanes engine initialized successfully");
  LOG_INFO("[onnxrt_engine] - Input: %dx%d", model_input_width_, model_input_height_);
  LOG_INFO("[onnxrt_engine] - Output: %dx%d (3 channels)", model_output_width_, model_output_height_);
}

EgoLanesOnnxEngine::~EgoLanesOnnxEngine()
{
  // Smart pointers handle cleanup automatically
}

void EgoLanesOnnxEngine::preprocessEgoLanes(const cv::Mat& input_image, float* buffer)
{
  // Step 1: Resize to model input size (320x640)
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(model_input_width_, model_input_height_), 
             0, 0, cv::INTER_LINEAR);
  
  // Step 2: Convert BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
  
  // Step 3: Convert to float and normalize to [0, 1]
  cv::Mat float_image;
  rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
  
  // Step 4: Apply ImageNet normalization and convert HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  int channel_size = model_input_height_ * model_input_width_;
  
  // Normalize each channel with ImageNet stats and copy to buffer
  for (int c = 0; c < 3; ++c) {
    float* channel_buffer = buffer + c * channel_size;
    float* channel_data = reinterpret_cast<float*>(channels[c].data);
    
    for (int i = 0; i < channel_size; ++i) {
      channel_buffer[i] = (channel_data[i] - MEAN[c]) / STD[c];
    }
  }
}

bool EgoLanesOnnxEngine::doInference(const cv::Mat& input_image)
{
  // Preprocess image
  preprocessEgoLanes(input_image, input_buffer_.data());
  
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
  
  return true;
}

LaneSegmentation EgoLanesOnnxEngine::inference(
  const cv::Mat& input_image,
  float threshold)
{
  // Run inference
  if (!doInference(input_image)) {
    LOG_ERROR("[onnxrt_engine] Inference failed");
    return LaneSegmentation{};
  }
  
  // Post-process and return lane masks
  return postProcess(threshold);
}

LaneSegmentation EgoLanesOnnxEngine::postProcess(float threshold)
{
  LaneSegmentation result;
  result.height = model_output_height_;
  result.width = model_output_width_;
  
  if (output_tensors_.empty()) {
    LOG_ERROR("[onnxrt_engine] No output tensors available");
    return result;
  }
  
  const float* raw_output = output_tensors_[0].GetTensorData<float>();
  
  // Output format: [1, 3, height, width] in CHW format
  int channel_size = model_output_height_ * model_output_width_;
  
  // Initialize output masks
  result.ego_left = cv::Mat(model_output_height_, model_output_width_, CV_32FC1);
  result.ego_right = cv::Mat(model_output_height_, model_output_width_, CV_32FC1);
  result.other_lanes = cv::Mat(model_output_height_, model_output_width_, CV_32FC1);
  
  // Extract and threshold each channel
  // Channel 0: Ego left lane
  const float* channel_0 = raw_output;
  for (int i = 0; i < channel_size; ++i) {
    result.ego_left.at<float>(i) = (channel_0[i] > threshold) ? 1.0f : 0.0f;
  }
  
  // Channel 1: Ego right lane
  const float* channel_1 = raw_output + channel_size;
  for (int i = 0; i < channel_size; ++i) {
    result.ego_right.at<float>(i) = (channel_1[i] > threshold) ? 1.0f : 0.0f;
  }
  
  // Channel 2: Other lanes
  const float* channel_2 = raw_output + 2 * channel_size;
  for (int i = 0; i < channel_size; ++i) {
    result.other_lanes.at<float>(i) = (channel_2[i] > threshold) ? 1.0f : 0.0f;
  }
  
  return result;
}

const float* EgoLanesOnnxEngine::getRawTensorData() const
{
  if (output_tensors_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call inference() first.");
  }
  return output_tensors_[0].GetTensorData<float>();
}

std::vector<int64_t> EgoLanesOnnxEngine::getTensorShape() const
{
  if (output_tensors_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call inference() first.");
  }
  return output_tensors_[0].GetTensorTypeAndShapeInfo().GetShape();
}

}  // namespace autoware_pov::vision::egolanes

