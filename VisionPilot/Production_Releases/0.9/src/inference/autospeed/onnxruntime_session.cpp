#include "onnxruntime_session.hpp"
#include "logging.hpp"
#include <filesystem>
#include <stdexcept>

namespace autoware_pov::vision::autospeed
{

int OnnxRuntimeSessionFactory::num_threads_ = 0;

void OnnxRuntimeSessionFactory::setNumThreads(int num_threads)
{
  num_threads_ = num_threads;
}

std::unique_ptr<Ort::Session> OnnxRuntimeSessionFactory::createSession(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  // Create ONNX Runtime environment (thread-safe singleton pattern)
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AutoSpeedOnnxRuntime");
  
  if (provider == "cpu") {
    LOG_INFO("[onnxrt] Creating CPU session for model: %s (threads: %d)", 
             model_path.c_str(), num_threads_);
    return createCPUSession(env, model_path);
  }
  else if (provider == "tensorrt") {
    LOG_INFO("[onnxrt] Creating TensorRT session (%s) for model: %s", 
             precision.c_str(), model_path.c_str());
    return createTensorRTSession(env, model_path, precision, device_id, cache_dir);
  }
  else {
    throw std::runtime_error("Unsupported provider: " + provider + 
                             ". Use 'cpu' or 'tensorrt'");
  }
}

std::unique_ptr<Ort::Session> OnnxRuntimeSessionFactory::createCPUSession(
  Ort::Env& env,
  const std::string& model_path)
{
  Ort::SessionOptions session_options;
  
  // Configure number of threads if specified
  if (num_threads_ > 0) {
    session_options.SetIntraOpNumThreads(num_threads_);
  }

  // Simple CPU configuration - let ONNX Runtime decide optimal settings
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  // Create and return session
  auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  
  LOG_INFO("[onnxrt] CPU session created successfully");
  return session;
}

std::unique_ptr<Ort::Session> OnnxRuntimeSessionFactory::createTensorRTSession(
  Ort::Env& env,
  const std::string& model_path,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  Ort::SessionOptions session_options;
  
  // Enable graph optimizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  // Create cache directory if it doesn't exist
  std::filesystem::create_directories(cache_dir);
  
  // Configure TensorRT Provider Options
  const auto& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* tensorrt_options;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
  
  // Create unique cache prefix for fp32 vs fp16
  std::string cache_prefix = "onnxrt_" + precision + "_";
  
  // Prepare option keys and values
  std::vector<const char*> option_keys = {
    "device_id",                        // GPU device ID
    "trt_max_workspace_size",           // 2GB workspace
    "trt_fp16_enable",                  // FP16 precision
    "trt_engine_cache_enable",          // Enable engine caching
    "trt_engine_cache_path",            // Cache directory
    "trt_engine_cache_prefix",          // Unique prefix for fp16/fp32
    "trt_timing_cache_enable",          // Enable timing cache
    "trt_timing_cache_path",            // Same as engine cache
    "trt_builder_optimization_level",   // Max optimization
    "trt_min_subgraph_size"             // Minimum subgraph size
  };
  
  // Set FP16 based on precision argument
  std::string fp16_flag = (precision == "fp16") ? "1" : "0";
  std::string device_id_str = std::to_string(device_id);
  
  std::vector<const char*> option_values = {
    device_id_str.c_str(),     // GPU device
    "2147483648",              // 2GB workspace
    fp16_flag.c_str(),         // FP16 enable/disable
    "1",                       // Enable engine cache
    cache_dir.c_str(),         // Cache path
    cache_prefix.c_str(),      // Unique prefix (fp16/fp32)
    "1",                       // Enable timing cache
    cache_dir.c_str(),         // Timing cache path
    "5",                       // Max optimization level
    "1"                        // Min subgraph size
  };
  
  // Update TensorRT options
  Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
    tensorrt_options,
    option_keys.data(),
    option_values.data(),
    option_keys.size()
  ));
  
  // Append TensorRT provider to session
  session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
  
  // Fallback to CUDA if TensorRT fails for any subgraph
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = device_id;
  session_options.AppendExecutionProvider_CUDA(cuda_options);
  
  // Create session
  auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  
  // Release TensorRT options
  api.ReleaseTensorRTProviderOptions(tensorrt_options);
  
  LOG_INFO("[onnxrt] TensorRT session created successfully");
  LOG_INFO("[onnxrt] - Precision: %s", precision.c_str());
  LOG_INFO("[onnxrt] - Device: %d", device_id);
  LOG_INFO("[onnxrt] - Cache: %s/%s*.engine", cache_dir.c_str(), cache_prefix.c_str());
  
  return session;
}

}  // namespace autoware_pov::vision::autospeed

