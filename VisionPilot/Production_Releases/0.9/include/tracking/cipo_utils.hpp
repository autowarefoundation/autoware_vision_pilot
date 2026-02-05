#pragma once

#include <vector>
#include <limits>
#include "object_finder.hpp"

namespace autoware_pov::vision {

/**
 * @brief Helper utilities for CIPO selection logic
 * 
 * Keeps the main ObjectFinder class clean by separating CIPO selection logic.
 */
class CIPOUtils {
public:
    /**
     * @brief Find the closest object of a specific class/level
     * @param tracked_objects List of all tracked objects
     * @param class_id The class/level to search for (1 or 2)
     * @return Index of closest object, or -1 if none found
     */
    static int findClosestByLevel(const std::vector<TrackedObject>& tracked_objects, 
                                   int class_id);
    
    /**
     * @brief Select the main CIPO from Level 1 and Level 2 candidates
     * @param tracked_objects List of all tracked objects
     * @param level1_idx Index of closest Level 1 object (-1 if none)
     * @param level2_idx Index of closest Level 2 object (-1 if none)
     * @return Index of the main CIPO (whichever is closer), or -1 if none
     */
    static int selectMainCIPO(const std::vector<TrackedObject>& tracked_objects,
                             int level1_idx, 
                             int level2_idx);
};

} // namespace autoware_pov::vision

