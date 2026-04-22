/** 
 * @file camera_calibration.hpp
 * @brief Camera calibration utilities for VisionPilot
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>


namespace autoware_pov::vision::camera {


    /**
     * @brief Camera intrinsics data structure
     * This structure holds intrinsic parameters of a camera, including:
     *  - Intrinsic matrix (K)
     *  - Distortion coefficients
     *  - Image dimensions
     */
    struct CameraIntrinsics {
        cv::Mat K;              // 3x3 intrinsic matrix
        cv::Mat dist_coeffs;    // 1x5 distortion coefficients
        int width;              // Image width
        int height;             // Image height
    };

    /**
     * @brief Camera extrinsics data structure
     * This structure holds extrinsic parameters of a camera, including:
     *  - Rotation angles (pitch, yaw, roll) in radians
     *  - Mount height in meters
     */
    struct CameraExtrinsics {
        double pitch_rad;           // Rotation around X-axis, in radians
        double yaw_rad;             // Rotation around Y-axis, in radians
        double roll_rad;            // Rotation around Z-axis, in radians
        double mount_height_m;      // Camera mount height, in meters
    };


    class CameraCalibration {

        public:

            /**
             * @brief Constructor inits static warp matrix initially.
             * Since mounting is firmly fixed, doing it once should be OK.
             */
            CameraCalibration(
                const CameraIntrinsics& inference_intrinsics,
                const CameraExtrinsics& inference_extrinsics,
                const CameraIntrinsics& standard_intrinsics,
                const CameraExtrinsics& standard_extrinsics
            );

            /**
             * @brief Low-latency main func to process each inference frame.
             * Applies perspective warp to convert from inference camera view
             * to standard camera view.
             */
            cv::Mat processFrame(
                const cv::Mat& raw_frame
            );

        private:

            cv::Mat K_inference_;       // Intrinsic matrix, inference camera
            cv::Mat dist_coeffs_;       // Distortion coefficients, inference camera
            cv::Mat H_warp_;            // Precomputed perspective warp matrix
            cv::Size target_size_;      // Standard pose image size (width, height)

            /**
             * @brief Compute perspective warp matrix (H) to transform from
             * inference camera view to standard camera view.
             */
            cv::Mat getRotationMatrix(
                double pitch_rad,
                double yaw_rad,
                double roll_rad
            );
    };

}   // namespace autoware_pov::vision::camera