#include <common/utils.hpp>
#include <image_preprocessing/image_preprocessor.hpp>
#include <image_preprocessing/vpi_warp.hpp>

#include <filesystem>

ImagePreprocessor::ImagePreprocessor(const Config& cfg)
    : cfg_(cfg)
{
    C_ = load_matrix("homography_C_matrix.yaml", "C");
}

void ImagePreprocessor::preprocess(const cv::Mat& image, cv::Mat& warped_image,
                                   cv::Mat& resized_image, const cv::Size& size) const
{
    // ── Warped path (AutoDrive): perspective transform → 1024×512 BEV ──
    vpiWarpPerspective(image, warped_image, C_, cv::Size(1024, 512),
                       cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

    // ── Resized path (AutoSteer / AutoSpeed): top-crop 2:1 → 1024×512 ──
    const int crop_top = compute_top_crop_2_1(image.rows, image.cols);
    const cv::Rect roi(0, crop_top, image.cols, image.rows - crop_top);
    const cv::Mat cropped = image(roi);
    cv::resize(cropped, resized_image, size, 0, 0, cv::INTER_LINEAR);
}
