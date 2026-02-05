#ifndef AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_SESSION_HPP_
#define AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_SESSION_HPP_

#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

namespace autoware_pov::vision::autospeed
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
   * @return Unique pointer to created session
   */
  static std::unique_ptr<Ort::Session> createSession(
    const std::string& model_path,
    const std::string& provider,
    const std::string& precision = "fp16",
    int device_id = 0,
    const std::string& cache_dir = "./trt_cache"
  );

  /**
   * @brief Set number of threads for CPU execution
   * 
   * @param num_threads Number of threads (0 = auto/default)
   */
  static void setNumThreads(int num_threads);

private:
  static int num_threads_;

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
   * - Optimal workspace size
   * - Timing cache
   */
  static std::unique_ptr<Ort::Session> createTensorRTSession(
    Ort::Env& env,
    const std::string& model_path,
    const std::string& precision,
    int device_id,
    const std::string& cache_dir
  );
};

}  // namespace autoware_pov::vision::autospeed

#endif  // AUTOWARE_POV_VISION_AUTOSPEED_ONNXRUNTIME_SESSION_HPP_

