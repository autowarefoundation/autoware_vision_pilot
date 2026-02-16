#ifndef AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_SESSION_HPP_
#define AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_SESSION_HPP_

#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

namespace autoware_pov::vision::egolanes
{

/**
 * @brief Session creation utilities for ONNX Runtime
 * 
 * This module handles all provider-specific configuration:
 * - CPU: Default provider
 * - TensorRT: Engine caching, FP16/FP32, optimization
 */
class OnnxRuntimeSessionFactory
{
public:
  /**
   * @brief Create ONNX Runtime session with specified provider
   * 
   * @param model_path Path to ONNX model file
   * @param provider "cpu" or "tensorrt"
   * @param precision "fp32" or "fp16" (only for TensorRT)
   * @param device_id GPU device ID (only for TensorRT)
   * @param cache_dir Directory for TensorRT engine cache (optional)
   * @param workspace_size_gb TensorRT workspace size in GB (default: 1GB)
   * @param cache_prefix Prefix for TensorRT engine cache files (default: "egolanes_")
   * @return Unique pointer to created session
   */
  static std::unique_ptr<Ort::Session> createSession(
    const std::string& model_path,
    const std::string& provider,
    const std::string& precision = "fp16",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache",
    double workspace_size_gb = 1.0,
    const std::string& cache_prefix = "egolanes_"
  );

private:
  /**
   * @brief Create CPU session with default options
   */
  static std::unique_ptr<Ort::Session> createCPUSession(
    Ort::Env& env,
    const std::string& model_path
  );

  /**
   * @brief Create TensorRT session with optimized settings
   * 
   * Enables:
   * - Engine caching for fast startup
   * - FP16/FP32 precision
   * - Configurable workspace size
   * - Timing cache
   */
  static std::unique_ptr<Ort::Session> createTensorRTSession(
    Ort::Env& env,
    const std::string& model_path,
    const std::string& precision,
    int device_id,
    const std::string& cache_dir,
    double workspace_size_gb,
    const std::string& cache_prefix
  );
};

}  // namespace autoware_pov::vision::egolanes

#endif  // AUTOWARE_POV_VISION_EGOLANES_ONNXRUNTIME_SESSION_HPP_



