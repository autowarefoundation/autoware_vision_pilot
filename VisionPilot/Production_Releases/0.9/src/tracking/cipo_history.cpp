#include "tracking/cipo_history.hpp"

namespace autoware_pov::vision::tracking {

CIPOHistory::CIPOHistory(size_t max_history_size) 
    : max_size_(max_history_size) {
}

void CIPOHistory::push(const CIPOSnapshot& snapshot) {
    // Add new snapshot to the back
    history_.push_back(snapshot);
    
    // Remove oldest entry if we exceed max size (circular buffer behavior)
    if (history_.size() > max_size_) {
        history_.pop_front();
    }
}

const CIPOSnapshot* CIPOHistory::getLatest() const {
    if (history_.empty()) {
        return nullptr;
    }
    return &history_.back();
}

const CIPOSnapshot* CIPOHistory::getPrevious() const {
    if (history_.size() < 2) {
        return nullptr;
    }
    return &history_[history_.size() - 2];
}

bool CIPOHistory::didCIPOChange() const {
    const CIPOSnapshot* latest = getLatest();
    const CIPOSnapshot* previous = getPrevious();
    
    // Can't determine change if we don't have at least 2 frames
    if (!latest || !previous) {
        return false;
    }
    
    // CIPO changed if track_id is different
    return latest->track_id != previous->track_id;
}

void CIPOHistory::clear() {
    history_.clear();
}

}  // namespace autoware_pov::vision::tracking

