#include "inference/onnxruntime_session.hpp"
#include <filesystem>
#include <stdexcept>
#include <cstdio>

// Simple logging macros (standalone version)
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

std::unique_ptr<Ort::Session> OnnxRuntimeSessionFactory::createSession(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir,
  double workspace_size_gb,
  const std::string& cache_prefix)
{
  // Create ONNX Runtime environment (thread-safe singleton pattern)
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EgoLanesOnnxRuntime");
  
  if (provider == "cpu") {
    LOG_INFO("[onnxrt] Creating CPU session for model: %s", model_path.c_str());
    return createCPUSession(env, model_path);
  }
  else if (provider == "tensorrt") {
    LOG_INFO("[onnxrt] Creating TensorRT session (%s) for model: %s", 
             precision.c_str(), model_path.c_str());
    return createTensorRTSession(env, model_path, precision, device_id, cache_dir, workspace_size_gb, cache_prefix);
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
  const std::string& cache_dir,
  double workspace_size_gb,
  const std::string& cache_prefix)
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
  std::string full_cache_prefix = cache_prefix + precision + "_";
  
  // Calculate workspace size in bytes
  size_t workspace_size_bytes = static_cast<size_t>(workspace_size_gb * 1024.0 * 1024.0 * 1024.0);
  std::string workspace_size_str = std::to_string(workspace_size_bytes);
  
  // Prepare option keys and values
  std::vector<const char*> option_keys = {
    "device_id",                        // GPU device ID
    "trt_max_workspace_size",           // Configurable workspace size
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
    workspace_size_str.c_str(), // Configurable workspace size (in bytes)
    fp16_flag.c_str(),         // FP16 enable/disable
    "1",                       // Enable engine cache
    cache_dir.c_str(),         // Cache path
    full_cache_prefix.c_str(), // Unique prefix (fp16/fp32)
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
  LOG_INFO("[onnxrt] - Workspace: %.1f GB", workspace_size_gb);
  LOG_INFO("[onnxrt] - Cache: %s/%s*.engine", cache_dir.c_str(), full_cache_prefix.c_str());
  
  return session;
}

}  // namespace autoware_pov::vision::egolanes

