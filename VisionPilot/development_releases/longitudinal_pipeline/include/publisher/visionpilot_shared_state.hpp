/**
 * @file visionpilot_shared_state.hpp
 * @brief POSIX shared-memory IPC for VisionPilot v0.9 control outputs.
 *
 * Exposes a single named shared-memory segment that carries the complete
 * per-frame outputs of both pipelines so any downstream consumer
 * (e.g. a CAN gateway, safety monitor, or ROS2 bridge) can read them
 * without touching the inference process.
 *
 * Thread / process safety
 * -----------------------
 * A seqlock is used so readers are always wait-free:
 *   - Writer increments seq to ODD before writing, then to EVEN after.
 *   - Reader retries if it observes an odd seq or if seq changed mid-read.
 * Only one writer is assumed (the VisionPilot inference process).
 *
 * Layout of the shared segment
 * -----------------------------
 *   [ VisionPilotState (POD) ]   sizeof(VisionPilotState)
 *
 * Shared-memory name: "/visionpilot_state"  (default, override via ctor)
 */

#pragma once

#include <cstdint>
#include <cstdbool>

namespace autoware_pov::vision::publisher {

// ============================================================
// Shared state struct — must remain POD (no std::atomic, no vtable)
// ============================================================
struct VisionPilotState {

    // ---- seqlock counter (64-bit; odd = write in progress) ----
    volatile uint64_t seq;

    // ---- frame metadata ----
    uint64_t frame_number;         // Monotonically increasing frame id

    // ========================================================
    // LATERAL outputs  (EgoLanes → AutoSteer → PathFinder → PID)
    // ========================================================
    double steering_pid_deg;       // Final filtered PID steering angle (deg)
    double steering_pid_raw_deg;   // Raw (unfiltered) PID output (deg)
    double steering_autosteer_deg; // AutoSteer model prediction (deg)
    bool   autosteer_valid;        // False on first frame (temporal buffer not ready)

    double cte_m;                  // Cross-track error from PathFinder (m)
    double yaw_error_rad;          // Yaw error from PathFinder (rad)
    double curvature_inv_m;        // Road curvature estimate (1/m)
    bool   path_valid;             // True when PathFinder has a fused estimate

    bool   lane_departure_warning; // True when drift > 50% of lane half-width

    // ========================================================
    // LONGITUDINAL outputs  (AutoSpeed → ObjectFinder → SpeedPlanner → PID)
    // ========================================================

    // CIPO (Closest In-Path Object)
    bool   cipo_exists;            // True when a valid CIPO is tracked
    int    cipo_track_id;          // Kalman track ID of the active CIPO
    int    cipo_class_id;          // 1 = Level-1, 2 = Level-2
    double cipo_distance_m;        // Filtered distance to CIPO (m)
    double cipo_velocity_ms;       // Relative velocity of CIPO (m/s, - = opening)
    bool   cut_in_detected;        // True on the frame a cut-in was detected
    bool   kalman_reset;           // True when Kalman state was forcibly reset

    // Speed planner
    double ideal_speed_ms;         // Commanded set-speed from RSS planner (m/s)
    double safe_distance_m;        // RSS d_min (m); 0 when no CIPO
    bool   fcw_active;             // Forward Collision Warning
    bool   aeb_active;             // Automatic Emergency Braking

    // Longitudinal PID controller
    double control_effort_ms2;     // Accel/decel command (m/s², + = accel, − = brake)

    // ========================================================
    // CAN / ego state (from CAN bus; NaN when unavailable)
    // ========================================================
    double ego_speed_ms;           // Ego vehicle speed (m/s)
    double ego_steering_angle_deg; // Measured steering angle (deg)
    bool   can_valid;              // True when CAN data is fresh
};

// ============================================================
// IPC writer/reader class
// ============================================================
class VisionPilotSharedState
{
public:
    /**
     * @brief Open (or create) the shared-memory segment.
     * @param name  POSIX shm name, e.g. "/visionpilot_state"
     * @param owner True  → create + size the segment (inference process).
     *              False → attach read-only (consumer process).
     */
    explicit VisionPilotSharedState(const char* name = "/visionpilot_state",
                                    bool owner = true);
    ~VisionPilotSharedState();

    /**
     * @brief Write a complete state snapshot (seqlock protected).
     *  Only call from the inference process (owner=true).
     */
    void publish(const VisionPilotState& state);

    /**
     * @brief Read a consistent snapshot (seqlock retry loop).
     *  Safe to call from any reader process.
     * @param[out] out  Populated with the latest consistent state.
     */
    void read(VisionPilotState& out) const;

    /**
     * @brief Direct pointer — use only if you manage synchronisation yourself.
     */
    VisionPilotState* raw() const;

private:
    const char* name_;
    int         fd_;
    void*       ptr_;
    bool        owner_;
};

} // namespace autoware_pov::vision::publisher
