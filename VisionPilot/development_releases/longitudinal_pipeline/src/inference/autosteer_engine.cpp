#include "inference/autosteer_engine.hpp"
#include "inference/onnxruntime_session.hpp"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

// Simple logging macros (standalone version)
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

AutoSteerOnnxEngine::AutoSteerOnnxEngine(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  // Create session using factory (reuses same logic as EgoLanes)
  // Use 1GB workspace for AutoSteer with separate cache prefix
  session_ = OnnxRuntimeSessionFactory::createSession(
    model_path, provider, precision, device_id, cache_dir, 1.0, "autosteer_"
  );
  
  // Create memory info for CPU tensors
  memory_info_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
  );
  
  // Get input/output names and store them permanently
  Ort::AllocatorWithDefaultOptions allocator;
  
  // Input (typically "input")
  auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
  input_name_storage_ = std::string(input_name_allocated.get());
  input_names_.push_back(input_name_storage_.c_str());
  
  // Outputs: AutoSteer has 2 outputs (prev_steering, current_steering)
  // Reserve space first to avoid reallocation
  output_names_storage_.reserve(2);
  
  // Get and store both output names
  auto output_name_0_allocated = session_->GetOutputNameAllocated(0, allocator);
  output_names_storage_.emplace_back(output_name_0_allocated.get());
  
  auto output_name_1_allocated = session_->GetOutputNameAllocated(1, allocator);
  output_names_storage_.emplace_back(output_name_1_allocated.get());
  
  // Now get the c_str() pointers (after strings are stored and vector won't reallocate)
  output_names_.clear();
  output_names_.reserve(2);
  output_names_.push_back(output_names_storage_[0].c_str());
  output_names_.push_back(output_names_storage_[1].c_str());
  
  // Keep the old output_name_storage_ for backward compatibility (points to second output)
  output_name_storage_ = output_names_storage_[1];
  
  LOG_INFO("[autosteer_engine] Output 0 name: '%s'", output_names_storage_[0].c_str());
  LOG_INFO("[autosteer_engine] Output 1 name: '%s' (USED)", output_names_storage_[1].c_str());
  
  // Verify output names are not empty
  if (output_names_storage_[0].empty() || output_names_storage_[1].empty()) {
    LOG_ERROR("[autosteer_engine] ERROR: Output names are empty! Output 0: '%s', Output 1: '%s'", 
              output_names_storage_[0].c_str(), output_names_storage_[1].c_str());
    throw std::runtime_error("AutoSteer output names are empty");
  }
  
  // Get input shape: [1, 6, 80, 160]
  auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_input_channels_ = static_cast<int>(input_shape[1]);  // 6
  model_input_height_ = static_cast<int>(input_shape[2]);    // 80
  model_input_width_ = static_cast<int>(input_shape[3]);     // 160
  
  // Get output shapes: model has 2 outputs [1, 61] each (or [61] in some exports)
  auto output_shape_0 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto output_shape_1 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
  
  // Handle both [1, 61] and [61] output shapes
  if (output_shape_1.size() == 2) {
    model_output_classes_ = static_cast<int>(output_shape_1[1]);  // [1, 61] -> use index 1
  } else if (output_shape_1.size() == 1) {
    model_output_classes_ = static_cast<int>(output_shape_1[0]);  // [61] -> use index 0
  } else {
    LOG_ERROR("[autosteer_engine] Unexpected output shape dimensions: %zu", output_shape_1.size());
    model_output_classes_ = 61;  // Fallback to expected value
  }
  
  LOG_INFO("[autosteer_engine] AutoSteer engine initialized successfully");
  LOG_INFO("[autosteer_engine] - Input: [1, %d, %d, %d]", 
           model_input_channels_, model_input_height_, model_input_width_);
  LOG_INFO("[autosteer_engine] - Output classes: %d (steering range: -30 to +30 deg)", model_output_classes_);
  LOG_INFO("[autosteer_engine] - Using output index 1 (current frame prediction)");
}

AutoSteerOnnxEngine::~AutoSteerOnnxEngine()
{
  // Smart pointers handle cleanup automatically
}

bool AutoSteerOnnxEngine::doInference(const std::vector<float>& input_buffer)
{
  // Validate input size
  size_t expected_size = model_input_channels_ * model_input_height_ * model_input_width_;
  if (input_buffer.size() != expected_size) {
    LOG_ERROR("[autosteer_engine] Invalid input size: %zu (expected %zu)", 
              input_buffer.size(), expected_size);
    return false;
  }
  
  // Create input tensor from pre-concatenated buffer
  std::vector<int64_t> input_shape = {1, model_input_channels_, model_input_height_, model_input_width_};
  auto input_tensor = Ort::Value::CreateTensor<float>(
    *memory_info_,
    const_cast<float*>(input_buffer.data()),  // ONNX Runtime requires non-const
    input_buffer.size(),
    input_shape.data(),
    input_shape.size()
  );
  
  // Run inference (ONNX Runtime allocates output automatically)
  // Note: AutoSteer model has 2 outputs (prev, current), we need both
  try {
    // Verify output names are valid before calling Run
    if (output_names_.size() < 2) {
      LOG_ERROR("[autosteer_engine] Not enough output names: %zu (expected 2)", output_names_.size());
      return false;
    }
    
    if (output_names_[0] == nullptr || output_names_[1] == nullptr) {
      LOG_ERROR("[autosteer_engine] Output name pointers are null!");
      return false;
    }
    
    if (strlen(output_names_[0]) == 0 || strlen(output_names_[1]) == 0) {
      LOG_ERROR("[autosteer_engine] Output names are empty! Output 0: '%s', Output 1: '%s'", 
                output_names_[0] ? output_names_[0] : "null", 
                output_names_[1] ? output_names_[1] : "null");
      return false;
    }
    
    output_tensors_ = session_->Run(
      Ort::RunOptions{nullptr},
      input_names_.data(),
      &input_tensor,
      1,
      output_names_.data(),
      2  // Request BOTH outputs
    );
  } catch (const Ort::Exception& e) {
    LOG_ERROR("[autosteer_engine] Inference failed: %s", e.what());
    return false;
  }
  
  return true;
}

float AutoSteerOnnxEngine::postProcess()
{
  if (output_tensors_.size() < 2) {
    LOG_ERROR("[autosteer_engine] Expected 2 output tensors, got %zu", output_tensors_.size());
    if (output_tensors_.empty()) {
      return 0.0f;
    }
  }
  
  // AutoSteer model returns TWO outputs: (prev_frame_prediction, current_frame_prediction)
  // We use the SECOND output (index 1) for current frame steering
  int output_index = (output_tensors_.size() >= 2) ? 1 : 0;
  const float* raw_output = output_tensors_[output_index].GetTensorData<float>();
  
  // Find argmax (class with highest logit value)
  int max_class = 0;
  float max_value = raw_output[0];
  
  for (int i = 1; i < model_output_classes_; ++i) {
    if (raw_output[i] > max_value) {
      max_value = raw_output[i];
      max_class = i;
    }
  }
  
  // Convert class to steering angle: argmax - 30
  // Classes: 0-60 → Angles: -30 to +30 degrees
  float steering_angle = static_cast<float>(max_class - 30);
  
  return steering_angle;
}

float AutoSteerOnnxEngine::inference(const std::vector<float>& concat_input)
{
  // Run inference
  if (!doInference(concat_input)) {
    LOG_ERROR("[autosteer_engine] Inference failed");
    return 0.0f;
  }
  
  // Post-process and return steering angle
  return postProcess();
}

std::vector<float> AutoSteerOnnxEngine::getRawOutputLogits() const
{
  if (output_tensors_.size() < 2) {
    LOG_ERROR("[autosteer_engine] Expected 2 output tensors, got %zu", output_tensors_.size());
    if (output_tensors_.empty()) {
      return std::vector<float>();
    }
  }
  
  // Return the SECOND output (index 1), which is the current frame prediction
  int output_index = (output_tensors_.size() >= 2) ? 1 : 0;
  const float* raw_output = output_tensors_[output_index].GetTensorData<float>();
  return std::vector<float>(raw_output, raw_output + model_output_classes_);
}

}  // namespace autoware_pov::vision::egolanes

