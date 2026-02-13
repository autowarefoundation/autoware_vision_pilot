#include "../../include/tracking/cipo_utils.hpp"
#include <limits>

namespace autoware_pov::vision::tracking {

int CIPOUtils::findClosestByLevel(const std::vector<TrackedObject>& tracked_objects, 
                                   int class_id) {
    float min_distance = std::numeric_limits<float>::infinity();
    int closest_idx = -1;
    
    for (size_t i = 0; i < tracked_objects.size(); i++) {
        const auto& obj = tracked_objects[i];
        
        // Must match class and have valid distance
        if (obj.class_id == class_id && obj.distance_m > 0 && obj.distance_m < min_distance) {
            min_distance = obj.distance_m;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

int CIPOUtils::selectMainCIPO(const std::vector<TrackedObject>& tracked_objects,
                              int level1_idx, 
                              int level2_idx) {
    // No candidates
    if (level1_idx < 0 && level2_idx < 0) {
        return -1;
    }
    
    // Only one candidate
    if (level1_idx < 0) return level2_idx;
    if (level2_idx < 0) return level1_idx;
    
    // Both exist - pick closer one
    float dist1 = tracked_objects[level1_idx].distance_m;
    float dist2 = tracked_objects[level2_idx].distance_m;
    
    return (dist1 <= dist2) ? level1_idx : level2_idx;
}

} // namespace autoware_pov::vision::tracking

