#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

/**
 * Simple 1D Kalman Filter for Position-Only Measurement (POM) System
 * 
 * State vector: x = [position, velocity]^T
 * - position: longitudinal distance (world_point.x)
 * - velocity: rate of change of position
 * 
 * Dynamic Model (Constant Velocity):
 *   x_k = Φ * x_{k-1} + w_k
 *   where Φ = [[1, T],
 *              [0, 1]]
 * 
 * Measurement Model (Position-Only):
 *   z_k = H * x_k + v_k
 *   where H = [1, 0]
 */
class KalmanFilter {
public:
    /**
     * Constructor
     * @param process_noise_pos Process noise for position (Q[0][0])
     * @param process_noise_vel Process noise for velocity (Q[1][1])
     * @param measurement_noise Measurement noise for position (R)
     */
    KalmanFilter(float process_noise_pos = 1.0f, 
                 float process_noise_vel = 1.0f,
                 float measurement_noise = 1.0f);
    
    /**
     * Initialize the filter with first measurement
     * @param initial_position Initial measured position
     */
    void initialize(float initial_position);
    
    /**
     * Predict step: Predict state at time k using previous state
     * @param dt Time interval (T) between measurements
     */
    void predict(float dt);
    
    /**
     * Update step: Correct prediction using new measurement
     * @param measured_position New position measurement (z_k)
     */
    void update(float measured_position);
    
    /**
     * Get current estimated position
     */
    float getPosition() const { return x_[0]; }
    
    /**
     * Get current estimated velocity
     */
    float getVelocity() const { return x_[1]; }
    
    /**
     * Check if filter is initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * Reset the filter
     */
    void reset();

private:
    // State vector: [position, velocity]
    float x_[2];
    
    // Covariance matrix P (2x2)
    float P_[2][2];
    
    // Process noise covariance Q (2x2)
    float Q_[2][2];
    
    // Measurement noise covariance R (scalar for 1D)
    float R_;
    
    // Measurement matrix H = [1, 0]
    static constexpr float H_[2] = {1.0f, 0.0f};
    
    // Initialization flag
    bool initialized_;
};

#endif // KALMAN_FILTER_HPP

