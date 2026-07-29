#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ImagePreprocessor {
public:
    struct Config {
        bool use_vpi = false;  // true → VPI CUDA warp; false → OpenCV CPU warp
    };

    explicit ImagePreprocessor(const Config& cfg = {});

    void preprocess(const cv::Mat& image, cv::Mat& warped_image,
                    cv::Mat& resized_image, const cv::Size& size) const;

    const cv::Mat& C_mat() const { return C_; }

private:
    cv::Mat C_;
    Config  cfg_;
};

