#ifndef AUTOSPEED_DETECTION_HPP_
#define AUTOSPEED_DETECTION_HPP_

namespace autoware_pov::vision::autospeed
{

struct Detection {
  float x1, y1, x2, y2;  // Bounding box corners
  float confidence;       // Detection confidence
  int class_id;          // Object class ID
};

}  // namespace autoware_pov::vision::autospeed

#endif  // AUTOSPEED_DETECTION_HPP_
