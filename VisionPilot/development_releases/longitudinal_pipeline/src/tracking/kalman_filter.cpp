#include "../../include/tracking/kalman_filter.hpp"
#include <cmath>

KalmanFilter::KalmanFilter(float process_noise_pos, 
                           float process_noise_vel,
                           float measurement_noise) 
    : R_(measurement_noise), initialized_(false) {
    
    // Initialize state vector to zero
    x_[0] = 0.0f;  // position
    x_[1] = 0.0f;  // velocity
    
    // Initialize covariance matrix P with high uncertainty
    P_[0][0] = 1000.0f;  // position variance
    P_[0][1] = 0.0f;     // position-velocity covariance
    P_[1][0] = 0.0f;     // velocity-position covariance
    P_[1][1] = 1000.0f;  // velocity variance
    
    // Process noise covariance Q
    Q_[0][0] = process_noise_pos;  // position process noise
    Q_[0][1] = 0.0f;
    Q_[1][0] = 0.0f;
    Q_[1][1] = process_noise_vel;  // velocity process noise
}

void KalmanFilter::initialize(float initial_position) {
    // Set initial state
    x_[0] = initial_position;  // position
    x_[1] = 0.0f;              // velocity (unknown, assume 0)
    
    // Reset covariance
    P_[0][0] = 10.0f;   // Low uncertainty in position (we just measured it)
    P_[0][1] = 0.0f;
    P_[1][0] = 0.0f;
    P_[1][1] = 100.0f;  // High uncertainty in velocity (not measured)
    
    initialized_ = true;
}

void KalmanFilter::predict(float dt) {
    if (!initialized_) return;
    
    // ===== STEP 1: State Prediction =====
    // Equation E9: x̃_k = Φ * x̂_{k-1}
    // 
    // Φ = [[1, T],    x = [position]
    //      [0, 1]]        [velocity]
    //
    // x̃_k = [[1, T],  * [x_{k-1}]   = [x_{k-1} + T * v_{k-1}]
    //         [0, 1]]    [v_{k-1}]     [v_{k-1}             ]
    
    float x_pred[2];
    x_pred[0] = x_[0] + dt * x_[1];  // position = position + velocity * dt
    x_pred[1] = x_[1];                // velocity = velocity (constant velocity model)
    
    // ===== STEP 2: Covariance Prediction =====
    // Equation E12: P̃_k = Φ * P_{k-1} * Φ^T + Q
    //
    // First: Φ * P_{k-1}
    float PhiP[2][2];
    PhiP[0][0] = P_[0][0] + dt * P_[1][0];
    PhiP[0][1] = P_[0][1] + dt * P_[1][1];
    PhiP[1][0] = P_[1][0];
    PhiP[1][1] = P_[1][1];
    
    // Second: (Φ * P_{k-1}) * Φ^T
    float P_pred[2][2];
    P_pred[0][0] = PhiP[0][0] + dt * PhiP[0][1];
    P_pred[0][1] = PhiP[0][1];
    P_pred[1][0] = PhiP[1][0] + dt * PhiP[1][1];
    P_pred[1][1] = PhiP[1][1];
    
    // Third: Add process noise Q
    P_pred[0][0] += Q_[0][0];
    P_pred[0][1] += Q_[0][1];
    P_pred[1][0] += Q_[1][0];
    P_pred[1][1] += Q_[1][1];
    
    // Update state and covariance with predictions
    x_[0] = x_pred[0];
    x_[1] = x_pred[1];
    P_[0][0] = P_pred[0][0];
    P_[0][1] = P_pred[0][1];
    P_[1][0] = P_pred[1][0];
    P_[1][1] = P_pred[1][1];
}

void KalmanFilter::update(float measured_position) {
    if (!initialized_) {
        initialize(measured_position);
        return;
    }
    
    // ===== STEP 3: Innovation (Measurement Residual) =====
    // y_k = z_k - H * x̃_k
    // where z_k is the measurement and H = [1, 0]
    // 
    // y_k = z_k - (1 * position + 0 * velocity)
    //     = z_k - position
    
    float innovation = measured_position - x_[0];
    
    // ===== STEP 4: Innovation Covariance =====
    // S_k = H * P̃_k * H^T + R
    // where H = [1, 0]
    //
    // H * P̃_k = [1, 0] * [[P00, P01],  = [P00, P01]
    //                      [P10, P11]]
    //
    // (H * P̃_k) * H^T = [P00, P01] * [[1],  = P00
    //                                   [0]]
    //
    // S_k = P00 + R
    
    float innovation_cov = P_[0][0] + R_;
    
    // ===== STEP 5: Kalman Gain =====
    // Equation E11: K_k = P̃_k * H^T * (H * P̃_k * H^T + R)^{-1}
    //                   = P̃_k * H^T * S_k^{-1}
    //
    // P̃_k * H^T = [[P00, P01],  * [[1],  = [[P00],
    //               [P10, P11]]     [0]]     [P10]]
    //
    // K_k = [[P00],  * (1 / S_k) = [[P00 / S_k],
    //        [P10]]                 [P10 / S_k]]
    
    float K[2];
    K[0] = P_[0][0] / innovation_cov;  // Kalman gain for position
    K[1] = P_[1][0] / innovation_cov;  // Kalman gain for velocity
    
    // ===== STEP 6: State Update =====
    // Equation E10: x̂_k = x̃_k + K_k * (z_k - H * x̃_k)
    //                    = x̃_k + K_k * innovation
    //
    // [[position],  = [[position],  + [[K0],  * innovation
    //  [velocity]]     [velocity]]     [K1]]
    
    x_[0] = x_[0] + K[0] * innovation;  // Updated position
    x_[1] = x_[1] + K[1] * innovation;  // Updated velocity
    
    // ===== STEP 7: Covariance Update =====
    // P_k = (I - K_k * H) * P̃_k
    // where I is identity matrix and H = [1, 0]
    //
    // K_k * H = [[K0],  * [1, 0] = [[K0, 0],
    //            [K1]]               [K1, 0]]
    //
    // I - K_k * H = [[1, 0],  - [[K0, 0],  = [[1-K0, 0],
    //                [0, 1]]     [K1, 0]]     [-K1,  1]]
    //
    // (I - K_k * H) * P̃_k = [[1-K0, 0],  * [[P00, P01],
    //                         [-K1,  1]]     [P10, P11]]
    
    float P_new[2][2];
    P_new[0][0] = (1.0f - K[0]) * P_[0][0];
    P_new[0][1] = (1.0f - K[0]) * P_[0][1];
    P_new[1][0] = -K[1] * P_[0][0] + P_[1][0];
    P_new[1][1] = -K[1] * P_[0][1] + P_[1][1];
    
    // Update covariance
    P_[0][0] = P_new[0][0];
    P_[0][1] = P_new[0][1];
    P_[1][0] = P_new[1][0];
    P_[1][1] = P_new[1][1];
}

void KalmanFilter::reset() {
    initialized_ = false;
    x_[0] = 0.0f;
    x_[1] = 0.0f;
    P_[0][0] = 1000.0f;
    P_[0][1] = 0.0f;
    P_[1][0] = 0.0f;
    P_[1][1] = 1000.0f;
}

