/**
 * @file estimator.cpp
 * @brief Bayes filter implementation
 * 
 * Ported from PATHFINDER ROS2 package
 */

#include "path_planning/estimator.hpp"

void Estimator::initialize(const std::array<Gaussian, STATE_DIM> &init_state)
{
    state = init_state;
}

void Estimator::predict(const std::array<Gaussian, STATE_DIM> &process)
{
    for (size_t i = 0; i < STATE_DIM; ++i)
    {
        state[i].mean += process[i].mean;         // Apply delta (e.g., motion model)
        state[i].variance += process[i].variance; // Add process noise
    }
}

void Estimator::update(const std::array<Gaussian, STATE_DIM> &measurement)
{
    // Update step (product of Gaussians)
    for (size_t i = 0; i < STATE_DIM; ++i)
    {
        double v0 = state[i].variance;
        double m0 = state[i].mean;

        double v1, m1;
        if (std::isnan(measurement[i].mean))
        {
            state[i].variance *= 1.25;
            continue;
        }
        else
        {
            v1 = measurement[i].variance;
            m1 = measurement[i].mean;
        }

        double v2 = (v0 * v1) / (v0 + v1);
        double m2 = (m0 * v1 + m1 * v0) / (v0 + v1);

        state[i] = {m2, v2};
    }

    // Perform fusion
    for (const auto &[start_idx, end_idx] : fusion_rules)
    {
        double inv_var_sum = 0.0;
        double weighted_mean_sum = 0.0;

        for (size_t i = start_idx; i < end_idx; ++i)
        {
            const auto &g = state[i];
            if (g.variance <= 0.0)
                continue;

            inv_var_sum += 1.0 / g.variance;
            weighted_mean_sum += g.mean / g.variance;
        }

        if (inv_var_sum > 0.0)
        {
            double fused_var = 1.0 / inv_var_sum;
            double fused_mean = fused_var * weighted_mean_sum;
            if (end_idx < STATE_DIM)
                state[end_idx] = {fused_mean, fused_var};
        }
    }
}

void Estimator::configureFusionGroups(const std::vector<std::pair<size_t, size_t>> &rules)
{
    fusion_rules = rules;
}

const std::array<Gaussian, STATE_DIM> &Estimator::getState() const
{
    return state;
}

