#ifndef DRAW_DET_HPP_
#define DRAW_DET_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include "../tracking/object_finder.hpp"
#include "../inference/autospeed/onnxruntime_engine.hpp"

// Visualization helper - draws detections and tracked objects with IDs and CIPO indicator
void drawTrackedObjects(cv::Mat& frame,
                        const std::vector<autoware_pov::vision::autospeed::Detection>& detections,
                        const std::vector<autoware_pov::vision::tracking::TrackedObject>& tracked_objects,
                        const autoware_pov::vision::tracking::CIPOInfo& cipo,
                        bool cut_in_detected = false,
                        bool kalman_reset = false);

#endif // DRAW_DET_HPP_
