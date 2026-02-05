#include "tensorrt_engine.hpp"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(status)                         \
  do {                                             \
    auto ret = (status);                           \
    if (ret != 0) {                                \
      LOG_ERROR(                                   \
        "[autospeed_trt] CUDA failure: %s",       \
        cudaGetErrorString(ret));                  \
      throw std::runtime_error("CUDA failure");    \
    }                                              \
  } while (0)

namespace autoware_pov::vision::autospeed
{

void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      LOG_ERROR("[autospeed_trt] %s", msg);
    } else if (severity == Severity::kWARNING) {
      LOG_WARN("[autospeed_trt] %s", msg);
    } else {
      LOG_INFO("[autospeed_trt] %s", msg);
    }
  }
}

AutoSpeedTensorRTEngine::AutoSpeedTensorRTEngine(
  const std::string & model_path, 
  const std::string & precision, 
  int gpu_id)
{
  CUDA_CHECK(cudaSetDevice(gpu_id));

  std::string onnx_path = model_path;
  
  // Check if input is PyTorch checkpoint
  if (model_path.substr(model_path.find_last_of(".") + 1) == "pt") {
    LOG_INFO("[autospeed_trt] Detected PyTorch checkpoint: %s", model_path.c_str());
    onnx_path = model_path.substr(0, model_path.find_last_of(".")) + ".onnx";
    
    // Check if ONNX already exists
    std::ifstream onnx_check(onnx_path);
    if (!onnx_check) {
      LOG_INFO("[autospeed_trt] Converting PyTorch to ONNX...");
      convertPyTorchToOnnx(model_path, onnx_path);
    } else {
      LOG_INFO("[autospeed_trt] Found existing ONNX model: %s", onnx_path.c_str());
    }
  }

  // Now handle ONNX → TensorRT engine
  std::string engine_path = onnx_path + "." + precision + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file) {
    LOG_INFO("[autospeed_trt] Found pre-built %s engine at %s", precision.c_str(), engine_path.c_str());
    loadEngine(engine_path);
  } else {
    LOG_INFO("[autospeed_trt] No pre-built %s engine found. Building from ONNX: %s", 
             precision.c_str(), onnx_path.c_str());
    buildEngineFromOnnx(onnx_path, precision);
    
    LOG_INFO("[autospeed_trt] Saving %s engine to %s", precision.c_str(), engine_path.c_str());
    std::unique_ptr<nvinfer1::IHostMemory> model_stream{engine_->serialize()};
    std::ofstream out_file(engine_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char *>(model_stream->data()), model_stream->size());
  }
  
  // Create execution context
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  stream_ = stream;
  
  // Get input/output tensor information
  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);

  auto input_dims = engine_->getTensorShape(input_name);
  auto output_dims = engine_->getTensorShape(output_name);

  model_input_height_ = input_dims.d[2];
  model_input_width_ = input_dims.d[3];
  
  // AutoSpeed output format: [1, num_attributes, num_predictions]
  // e.g., [1, 8, 8400] for YOLO-like models (4 bbox + 4 classes, 8400 anchors)
  model_output_channels_ = output_dims.d[1];      // 8 (4 bbox + 4 classes)
  model_output_predictions_ = output_dims.d[2];   // 8400 (number of predictions)
  
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1LL, std::multiplies<int64_t>());
  model_output_elem_count_ = output_vol;

  // Allocate GPU buffers
  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));

  // Set tensor addresses
  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(output_name, output_buffer_gpu_);

  // Allocate host output buffer
  output_buffer_host_.resize(model_output_elem_count_);
  
  LOG_INFO("[autospeed_trt] Engine initialized successfully");
  LOG_INFO("[autospeed_trt] Input: %dx%d, Output: %d attributes x %d predictions", 
           model_input_width_, model_input_height_, 
           model_output_channels_, model_output_predictions_);
}

AutoSpeedTensorRTEngine::~AutoSpeedTensorRTEngine()
{
  cudaFree(input_buffer_gpu_);
  cudaFree(output_buffer_gpu_);
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
}

void AutoSpeedTensorRTEngine::convertPyTorchToOnnx(
  const std::string & pytorch_path, 
  const std::string & onnx_path)
{
  // Call Python script to convert PyTorch → ONNX
  // This assumes you have a conversion script in your project
  std::string convert_script = "python3 -c \"import torch; "
    "model = torch.load('" + pytorch_path + "', map_location='cpu', weights_only=False)['model']; "
    "dummy_input = torch.randn(1, 3, 640, 640); "
    "torch.onnx.export(model, dummy_input, '" + onnx_path + "', "
    "opset_version=11, input_names=['input'], output_names=['output'])\"";
  
  int result = std::system(convert_script.c_str());
  if (result != 0) {
    throw std::runtime_error("Failed to convert PyTorch to ONNX. Ensure PyTorch is installed.");
  }
  
  LOG_INFO("[autospeed_trt] Successfully converted PyTorch to ONNX: %s", onnx_path.c_str());
}

void AutoSpeedTensorRTEngine::buildEngineFromOnnx(
  const std::string & onnx_path, 
  const std::string & precision)
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
  
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

  // Set optimization profile for AutoSpeed (640x640 input)
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    LOG_INFO("[autospeed_trt] Building TensorRT engine with FP16 precision");
  } else {
    LOG_INFO("[autospeed_trt] Building TensorRT engine with FP32 precision");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  if (!engine_) {
    throw std::runtime_error("Failed to build TensorRT engine.");
  }
}

void AutoSpeedTensorRTEngine::loadEngine(const std::string & engine_path)
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

void AutoSpeedTensorRTEngine::preprocessAutoSpeed(const cv::Mat & input_image, float * buffer)
{
  // Store original dimensions for later coordinate transformation
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
  
  // Step 3: Convert to float and normalize to [0, 1] (NOT ImageNet normalization!)
  cv::Mat float_image;
  padded.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
  
  // Step 4: Convert BGR to RGB and HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  // BGR → RGB: Reverse channel order
  // AutoSpeed expects RGB, OpenCV loads as BGR
  int channel_size = target_h * target_w;
  memcpy(buffer, channels[2].data, channel_size * sizeof(float));  // R
  memcpy(buffer + channel_size, channels[1].data, channel_size * sizeof(float));  // G
  memcpy(buffer + 2 * channel_size, channels[0].data, channel_size * sizeof(float));  // B
}

bool AutoSpeedTensorRTEngine::doInference(const cv::Mat & input_image)
{
  // Allocate preprocessed data buffer
  std::vector<float> preprocessed_data(model_input_width_ * model_input_height_ * 3);
  
  // Preprocess with letterbox
  preprocessAutoSpeed(input_image, preprocessed_data.data());

  // Copy to GPU
  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, preprocessed_data.data(), preprocessed_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));

  // Run inference
  bool status = context_->enqueueV3(static_cast<cudaStream_t>(stream_));

  if (!status) {
    LOG_ERROR("[autospeed_trt] TensorRT inference failed");
    return false;
  }

  // Copy output back to host
  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, 
    static_cast<cudaStream_t>(stream_)));

  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  
  return true;
}

const float* AutoSpeedTensorRTEngine::getRawTensorData() const
{
  if (output_buffer_host_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call doInference() first.");
  }
  return output_buffer_host_.data();
}

std::vector<int64_t> AutoSpeedTensorRTEngine::getTensorShape() const
{
  // Return shape: [batch=1, num_predictions, num_attributes]
  return {1, static_cast<int64_t>(model_output_predictions_), static_cast<int64_t>(model_output_channels_)};
}

// High-level inference method (preprocessing + inference + post-processing)
std::vector<Detection> AutoSpeedTensorRTEngine::inference(
  const cv::Mat & input_image,
  float conf_thresh,
  float iou_thresh)
{
  // Run raw inference
  if (!doInference(input_image)) {
    LOG_ERROR("Inference failed");
    return {};
  }

  // Post-process and return detections
  return postProcess(conf_thresh, iou_thresh);
}

// Post-processing: parse raw output and transform to original image coordinates
std::vector<Detection> AutoSpeedTensorRTEngine::postProcess(float conf_thresh, float iou_thresh)
{
  std::vector<Detection> detections;
  
  const float* raw_output = output_buffer_host_.data();
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

// Compute IoU between two boxes
float AutoSpeedTensorRTEngine::computeIoU(const Detection& a, const Detection& b)
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

// Non-Maximum Suppression
std::vector<Detection> AutoSpeedTensorRTEngine::applyNMS(
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

}  // namespace autoware_pov::vision::autospeed

