#include "inference/tensorrt_autosteer_engine.hpp"
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(status)                         \
  do {                                             \
    auto ret = (status);                           \
    if (ret != 0) {                                \
      LOG_ERROR(                                   \
        "[tensorrt_autosteer] Cuda failure: %s",  \
        cudaGetErrorString(ret));                  \
      throw std::runtime_error("Cuda failure");    \
    }                                              \
  } while (0)

// Simple logging macros
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

void AutoSteerTensorRTLogger::log(Severity severity, const char* msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      LOG_ERROR("[tensorrt_autosteer] %s", msg);
    } else if (severity == Severity::kWARNING) {
      LOG_INFO("[tensorrt_autosteer] %s", msg);
    } else {
      LOG_INFO("[tensorrt_autosteer] %s", msg);
    }
  }
}

AutoSteerTensorRTEngine::AutoSteerTensorRTEngine(
  const std::string& model_path,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
  : cache_dir_(cache_dir),
    input_buffer_gpu_(nullptr),
    output_buffer_gpu_(nullptr),
    output_buffer_gpu_other_(nullptr),
    stream_(nullptr)
{
  CUDA_CHECK(cudaSetDevice(device_id));

  // Generate engine cache path: cache_dir/autosteer_<precision>.engine
  std::string model_name = model_path.substr(model_path.find_last_of("/\\") + 1);
  model_name = model_name.substr(0, model_name.find_last_of("."));  // Remove extension
  engine_cache_path_ = cache_dir + "/autosteer_" + model_name + "_" + precision + ".engine";

  // Try to load cached engine, otherwise build from ONNX
  std::ifstream engine_file(engine_cache_path_, std::ios::binary);
  if (engine_file) {
    LOG_INFO("[tensorrt_autosteer] Found pre-built %s engine at %s", precision.c_str(), engine_cache_path_.c_str());
    loadEngine(engine_cache_path_);
  } else {
    LOG_INFO("[tensorrt_autosteer] No pre-built %s engine found. Building from ONNX model: %s", 
             precision.c_str(), model_path.c_str());
    buildEngineFromOnnx(model_path, precision);
    
    LOG_INFO("[tensorrt_autosteer] Saving %s engine to %s", precision.c_str(), engine_cache_path_.c_str());
    std::unique_ptr<nvinfer1::IHostMemory> model_stream{engine_->serialize()};
    std::ofstream out_file(engine_cache_path_, std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
  }

  // Create execution context
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  // Create CUDA stream
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // Get input/output tensor names and shapes
  const char* input_name = engine_->getIOTensorName(0);
  
  // AutoSteer has 2 outputs - we need to set addresses for BOTH
  // Output 0: past frame prediction
  // Output 1: current frame prediction (we use this one)
  int num_tensors = engine_->getNbIOTensors();
  if (num_tensors < 3) {  // 1 input + 2 outputs
    throw std::runtime_error("AutoSteer model must have 1 input and 2 outputs");
  }
  
  // Find all output tensors and identify which is which
  std::vector<const char*> output_names;
  std::vector<std::string> output_name_strings;
  const char* primary_output_name = nullptr;   // Current frame prediction (output index 1, typically "101")
  const char* other_output_name = nullptr;     // Past frame prediction (output index 0)
  
  // Collect all outputs
  for (int i = 0; i < num_tensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (name != input_name) {
      output_names.push_back(name);
      output_name_strings.push_back(std::string(name));
      LOG_INFO("[tensorrt_autosteer] Found output %zu: '%s'", output_names.size() - 1, name);
    }
  }
  
  if (output_names.size() < 2) {
    throw std::runtime_error("AutoSteer model must have at least 2 outputs");
  }
  
  // Identify outputs: look for "101" (current frame) or use second output by position
  // In ONNX Runtime, output index 1 is the current frame prediction
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::string name_str = output_name_strings[i];
    if (name_str.find("101") != std::string::npos) {
      // Found "101" - this is the current frame prediction
      primary_output_name = output_names[i];
      output_tensor_name_ = name_str;
      other_output_name = (i == 0) ? output_names[1] : output_names[0];
      LOG_INFO("[tensorrt_autosteer] Identified '101' as current frame prediction (index %zu)", i);
      break;
    }
  }
  
  // If we didn't find "101", assume second output (index 1) is current frame prediction
  // This matches ONNX Runtime behavior where output index 1 is current frame
  if (!primary_output_name) {
    if (output_names.size() >= 2) {
      primary_output_name = output_names[1];  // Second output = current frame
      output_tensor_name_ = output_name_strings[1];
      other_output_name = output_names[0];    // First output = past frame
      LOG_INFO("[tensorrt_autosteer] Using output index 1 as current frame prediction (no '101' found)");
    } else {
      throw std::runtime_error("Failed to identify current frame output tensor");
    }
  }

  auto input_dims = engine_->getTensorShape(input_name);
  auto primary_output_dims = engine_->getTensorShape(primary_output_name);

  model_input_channels_ = input_dims.d[1];  // [1, 6, 80, 160]
  model_input_height_ = input_dims.d[2];
  model_input_width_ = input_dims.d[3];
  
  // Handle both [1, 61] and [61] output shapes
  if (primary_output_dims.nbDims == 2) {
    model_output_classes_ = primary_output_dims.d[1];  // [1, 61]
  } else if (primary_output_dims.nbDims == 1) {
    model_output_classes_ = primary_output_dims.d[0];  // [61]
  } else {
    LOG_ERROR("[tensorrt_autosteer] Unexpected output shape dimensions: %d", primary_output_dims.nbDims);
    model_output_classes_ = 61;  // Fallback to expected value
  }

  // Allocate GPU buffers
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = model_output_classes_;  // Each output: 61 classes

  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));
  
  // Allocate buffer for the other output (we don't use it, but TensorRT requires it)
  // Get dimensions of the other output (past frame prediction)
  auto other_output_dims = engine_->getTensorShape(other_output_name);
  int64_t other_output_vol = 61;  // Assume same size
  if (other_output_dims.nbDims == 2) {
    other_output_vol = other_output_dims.d[1];
  } else if (other_output_dims.nbDims == 1) {
    other_output_vol = other_output_dims.d[0];
  }
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_other_, other_output_vol * sizeof(float)));

  // Set tensor addresses for ALL tensors (TensorRT requires this)
  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(primary_output_name, output_buffer_gpu_);
  context_->setTensorAddress(other_output_name, output_buffer_gpu_other_);
  
  LOG_INFO("[tensorrt_autosteer] Found %zu outputs: %s (using), %s (unused)", 
           output_names.size(), output_tensor_name_.c_str(), other_output_name);

  // Allocate host output buffer
  output_buffer_host_.resize(output_vol);

  LOG_INFO("[tensorrt_autosteer] AutoSteer TensorRT engine initialized successfully");
  LOG_INFO("[tensorrt_autosteer] - Input: [1, %d, %d, %d]", 
           model_input_channels_, model_input_height_, model_input_width_);
  LOG_INFO("[tensorrt_autosteer] - Output classes: %d (steering range: -30 to +30 deg)", model_output_classes_);
  LOG_INFO("[tensorrt_autosteer] - Using output: %s", output_tensor_name_.c_str());
}

AutoSteerTensorRTEngine::~AutoSteerTensorRTEngine()
{
  if (input_buffer_gpu_) {
    cudaFree(input_buffer_gpu_);
  }
  if (output_buffer_gpu_) {
    cudaFree(output_buffer_gpu_);
  }
  if (output_buffer_gpu_other_) {
    cudaFree(output_buffer_gpu_other_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void AutoSteerTensorRTEngine::buildEngineFromOnnx(
  const std::string& onnx_path, const std::string& precision)
{
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  
  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
    std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  
  auto parser =
    std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file.");
  }
  
  // Set workspace size (1GB)
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

  // Set optimization profile for AutoSteer: [1, 6, 80, 160]
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  const char* input_name = network->getInput(0)->getName();
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 6, 80, 160));
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 6, 80, 160));
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 6, 80, 160));
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    LOG_INFO("[tensorrt_autosteer] Building TensorRT engine with FP16 precision");
  } else {
    LOG_INFO("[tensorrt_autosteer] Building TensorRT engine with FP32 precision");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  if (!engine_) {
    throw std::runtime_error("Failed to build TensorRT engine.");
  }
}

void AutoSteerTensorRTEngine::loadEngine(const std::string& engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!engine_) {
    throw std::runtime_error("Failed to load TensorRT engine.");
  }
}

bool AutoSteerTensorRTEngine::doInference(const std::vector<float>& input_buffer)
{
  // Validate input size
  size_t expected_size = model_input_channels_ * model_input_height_ * model_input_width_;
  if (input_buffer.size() != expected_size) {
    LOG_ERROR("[tensorrt_autosteer] Invalid input size: %zu (expected %zu)", 
              input_buffer.size(), expected_size);
    return false;
  }

  // Copy to GPU
  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, input_buffer.data(), input_buffer.size() * sizeof(float),
    cudaMemcpyHostToDevice, stream_));

  // Run inference
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    LOG_ERROR("[tensorrt_autosteer] TensorRT inference failed");
    return false;
  }

  // Copy output back to host
  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  
  return true;
}

float AutoSteerTensorRTEngine::postProcess()
{
  if (output_buffer_host_.empty()) {
    LOG_ERROR("[tensorrt_autosteer] No output data available");
    return 0.0f;
  }
  
  const float* raw_output = output_buffer_host_.data();
  
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

float AutoSteerTensorRTEngine::inference(const std::vector<float>& concat_input)
{
  // Run inference
  if (!doInference(concat_input)) {
    LOG_ERROR("[tensorrt_autosteer] Inference failed");
    return 0.0f;
  }
  
  // Post-process and return steering angle
  return postProcess();
}

std::vector<float> AutoSteerTensorRTEngine::getRawOutputLogits() const
{
  if (output_buffer_host_.empty()) {
    return std::vector<float>();
  }
  
  const float* raw_output = output_buffer_host_.data();
  return std::vector<float>(raw_output, raw_output + model_output_classes_);
}

}  // namespace autoware_pov::vision::egolanes

