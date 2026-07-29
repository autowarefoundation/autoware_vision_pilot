#pragma once

#include <opencv2/core.hpp>

// VPI-accelerated perspective warp.
// Compile with VPI_AVAILABLE to enable, otherwise falls back to cv::warpPerspective.
// VPI is part of NVIDIA JetPack — requires libvpi-dev.
//
// Usage:
//   vpiWarpPerspective(src, dst, H, {1024, 512}, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

void vpiWarpPerspective(const cv::Mat& src, cv::Mat& dst, const cv::Mat& H,
                        cv::Size dsize, int interp = cv::INTER_LINEAR,
                        int borderMode = cv::BORDER_REFLECT_101);
