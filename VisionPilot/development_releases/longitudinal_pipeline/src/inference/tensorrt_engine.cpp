#include "inference/tensorrt_engine.hpp"
#include "inference/lane_segmentation.hpp"
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(status)                         \
  do {                                             \
    auto ret = (status);                           \
    if (ret != 0) {                                \
      LOG_ERROR(                                   \
        "[tensorrt_engine] Cuda failure: %s",     \
        cudaGetErrorString(ret));                  \
      throw std::runtime_error("Cuda failure");    \
    }                                              \
  } while (0)

// Simple logging macros
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

void TensorRTLogger::log(Severity severity, const char* msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      LOG_ERROR("[tensorrt_engine] %s", msg);
    } else if (severity == Severity::kWARNING) {
      LOG_INFO("[tensorrt_engine] %s", msg);
    } else {
      LOG_INFO("[tensorrt_engine] %s", msg);
    }
  }
}

EgoLanesTensorRTEngine::EgoLanesTensorRTEngine(
  const std::string& model_path,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
  : cache_dir_(cache_dir),
    input_buffer_gpu_(nullptr),
    output_buffer_gpu_(nullptr),
    stream_(nullptr)
{
  CUDA_CHECK(cudaSetDevice(device_id));

  // Generate engine cache path: cache_dir/egolanes_<precision>.engine
  std::string model_name = model_path.substr(model_path.find_last_of("/\\") + 1);
  model_name = model_name.substr(0, model_name.find_last_of("."));  // Remove extension
  engine_cache_path_ = cache_dir + "/egolanes_" + model_name + "_" + precision + ".engine";

  // Try to load cached engine, otherwise build from ONNX
  std::ifstream engine_file(engine_cache_path_, std::ios::binary);
  if (engine_file) {
    LOG_INFO("[tensorrt_engine] Found pre-built %s engine at %s", precision.c_str(), engine_cache_path_.c_str());
    loadEngine(engine_cache_path_);
  } else {
    LOG_INFO("[tensorrt_engine] No pre-built %s engine found. Building from ONNX model: %s", 
             precision.c_str(), model_path.c_str());
    buildEngineFromOnnx(model_path, precision);
    
    LOG_INFO("[tensorrt_engine] Saving %s engine to %s", precision.c_str(), engine_cache_path_.c_str());
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
  const char* output_name = engine_->getIOTensorName(1);

  auto input_dims = engine_->getTensorShape(input_name);
  auto output_dims = engine_->getTensorShape(output_name);

  model_input_height_ = input_dims.d[2];  // NCHW: [1, 3, 320, 640]
  model_input_width_ = input_dims.d[3];
  
  model_output_channels_ = output_dims.d[1];  // [1, 3, 320, 640]
  model_output_height_ = output_dims.d[2];
  model_output_width_ = output_dims.d[3];

  // Allocate GPU buffers
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1LL, std::multiplies<int64_t>());

  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));

  // Set tensor addresses
  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(output_name, output_buffer_gpu_);

  // Allocate host output buffer
  output_buffer_host_.resize(output_vol);

  LOG_INFO("[tensorrt_engine] EgoLanes TensorRT engine initialized successfully");
  LOG_INFO("[tensorrt_engine] - Input: %dx%d", model_input_width_, model_input_height_);
  LOG_INFO("[tensorrt_engine] - Output: %dx%d (3 channels)", model_output_width_, model_output_height_);
}

EgoLanesTensorRTEngine::~EgoLanesTensorRTEngine()
{
  if (input_buffer_gpu_) {
    cudaFree(input_buffer_gpu_);
  }
  if (output_buffer_gpu_) {
    cudaFree(output_buffer_gpu_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void EgoLanesTensorRTEngine::buildEngineFromOnnx(
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

  // Set optimization profile for EgoLanes: [1, 3, 320, 640]
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  const char* input_name = network->getInput(0)->getName();
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 320, 640));
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 320, 640));
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 320, 640));
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    LOG_INFO("[tensorrt_engine] Building TensorRT engine with FP16 precision");
  } else {
    LOG_INFO("[tensorrt_engine] Building TensorRT engine with FP32 precision");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  if (!engine_) {
    throw std::runtime_error("Failed to build TensorRT engine.");
  }
}

void EgoLanesTensorRTEngine::loadEngine(const std::string& engine_path)
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

void EgoLanesTensorRTEngine::preprocessEgoLanes(const cv::Mat& input_image, float* buffer)
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

bool EgoLanesTensorRTEngine::doInference(const cv::Mat& input_image)
{
  // Preprocess image
  std::vector<float> preprocessed_data(model_input_width_ * model_input_height_ * 3);
  preprocessEgoLanes(input_image, preprocessed_data.data());

  // Copy to GPU
  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, preprocessed_data.data(), preprocessed_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, stream_));

  // Run inference
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    LOG_ERROR("[tensorrt_engine] TensorRT inference failed");
    return false;
  }

  // Copy output back to host
  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  
  return true;
}

LaneSegmentation EgoLanesTensorRTEngine::inference(
  const cv::Mat& input_image,
  float threshold)
{
  // Run inference
  if (!doInference(input_image)) {
    LOG_ERROR("[tensorrt_engine] Inference failed");
    return LaneSegmentation{};
  }
  
  // Post-process and return lane masks
  return postProcess(threshold);
}

LaneSegmentation EgoLanesTensorRTEngine::postProcess(float threshold)
{
  LaneSegmentation result;
  result.height = model_output_height_;
  result.width = model_output_width_;
  
  if (output_buffer_host_.empty()) {
    LOG_ERROR("[tensorrt_engine] No output data available");
    return result;
  }
  
  const float* raw_output = output_buffer_host_.data();
  
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

const float* EgoLanesTensorRTEngine::getRawTensorData() const
{
  if (output_buffer_host_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call inference() first.");
  }
  return output_buffer_host_.data();
}

std::vector<int64_t> EgoLanesTensorRTEngine::getTensorShape() const
{
  return {1, static_cast<int64_t>(model_output_channels_), 
          static_cast<int64_t>(model_output_height_), 
          static_cast<int64_t>(model_output_width_)};
}

}  // namespace autoware_pov::vision::egolanes

