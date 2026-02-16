#ifndef ZENOH_AUTOSPEED_PAYLOAD_HPP_
#define ZENOH_AUTOSPEED_PAYLOAD_HPP_

#include "autospeed/detection.hpp"
#include "object_finder.hpp"

namespace autoware_pov::vision::zenoh {

// Maximum counts for fixed-size payload
constexpr int MAX_DETECTIONS = 64;
constexpr int MAX_TRACKED = 32;

// Serializable tracked object (without Kalman filter)
struct TrackedObjectPayload {
    int track_id;
    int class_id;
    float confidence;
    int bbox[4];  // [x, y, w, h]
    float distance_m;
    float velocity_ms;

    TrackedObjectPayload() = default;

    TrackedObjectPayload(const TrackedObject& obj)
        : track_id(obj.track_id)
        , class_id(obj.class_id)
        , confidence(obj.confidence)
        , bbox{obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height}
        , distance_m(obj.distance_m)
        , velocity_ms(obj.velocity_ms)
    {}
};

// Unified AutoSpeed payload for Zenoh transport
// Single topic: autospeed/result
struct AutoSpeedPayload {
    // Detections (all classes, for visualization of class 3)
    int num_detections;
    autospeed::Detection detections[MAX_DETECTIONS];

    // Tracked objects (class 1, 2 with distance/velocity)
    int num_tracked;
    TrackedObjectPayload tracked[MAX_TRACKED];

    // CIPO + event flags
    CIPOInfo cipo;
    bool cut_in_detected;
    bool kalman_reset;

    AutoSpeedPayload()
        : num_detections(0)
        , num_tracked(0)
        , cut_in_detected(false)
        , kalman_reset(false)
    {}
};

}  // namespace autoware_pov::vision::zenoh

#endif  // ZENOH_AUTOSPEED_PAYLOAD_HPP_
