/**
 * @file estimator.hpp
 * @brief Bayes filter for temporal tracking of lane parameters
 * 
 * Ported from PATHFINDER ROS2 package (no ROS2 dependencies)
 */

#pragma once

#include <vector>
#include <array>
#include <cmath>

constexpr size_t STATE_DIM = 14;

struct Gaussian
{
    double mean;
    double variance;
};

class Estimator
{
private:
    std::array<Gaussian, STATE_DIM> state;
    std::vector<std::pair<size_t, size_t>> fusion_rules;

public:
    void initialize(const std::array<Gaussian, STATE_DIM> &init_state);
    void predict(const std::array<Gaussian, STATE_DIM> &process);
    void update(const std::array<Gaussian, STATE_DIM> &measurement);
    void configureFusionGroups(const std::vector<std::pair<size_t, size_t>> &rules);
    const std::array<Gaussian, STATE_DIM> &getState() const;
};

