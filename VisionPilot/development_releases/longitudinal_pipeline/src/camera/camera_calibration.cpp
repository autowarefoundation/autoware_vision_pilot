/**
 * @file camera_calibration.cpp
 * @brief Implementation of deterministic camera calibration and
 * perspective warping for E2E perception models.
 */

#include "camera/camera_calibration.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace autoware_pov::vision::camera {

    
    CameraCalibration::CameraCalibration(
        const CameraIntrinsics& inference_intrinsics,
        const CameraExtrinsics& inference_extrinsics,
        const CameraIntrinsics& standard_intrinsics,
        const CameraExtrinsics& standard_extrinsics
    ) {
        
        // 1. Store inference intrinsics
        K_inference_ = inference_intrinsics.K.clone();
        dist_coeffs_ = inference_intrinsics.dist_coeffs.clone();
        if (K_inference_.type() != CV_64F) {
            K_inference_.convertTo(K_inference_, CV_64F);  // Ensure double precision
        }
        if (dist_coeffs_.type() != CV_64F) {
            dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
        }
        target_size_ = cv::Size(standard_intrinsics.width, standard_intrinsics.height); // Target resolution

        // 2. Compute rotation matrices from extrinsics

        cv::Mat R_inf = getRotationMatrix(
            inference_extrinsics.pitch_rad,
            inference_extrinsics.yaw_rad,
            inference_extrinsics.roll_rad
        );
                                        
        cv::Mat R_std = getRotationMatrix(
            standard_extrinsics.pitch_rad,
            standard_extrinsics.yaw_rad,
            standard_extrinsics.roll_rad
        );

        // 3. Compute relative rotation: R_rel = R_std * R_inf^-1
        cv::Mat R_rel = R_std * R_inf.inv();

        // 4. Simulate vertical translation via focal length rescaling
        // Instead of planar homography, scale the target's vertical focal length to
        // mathematically squeeze image and correct depth estimation for E2E models.

        cv::Mat K_std_mod = standard_intrinsics.K.clone();
        if (K_std_mod.type() != CV_64F) {
            K_std_mod.convertTo(K_std_mod, CV_64F); // Ensure double precision
        }

        if (inference_extrinsics.mount_height_m <= 0.0) {
            throw std::invalid_argument("inference mount_height_m must be > 0");
        }
        double scale_factor = standard_extrinsics.mount_height_m / inference_extrinsics.mount_height_m;
        K_std_mod.at<double>(1, 1) *= scale_factor; // Scale f_y

        // 5. Compute master homography matrix
        // H_warp = K_std_mod * R_rel * K_inf^-1
        H_warp_ = K_std_mod * R_rel * K_inference_.inv();

    }

    cv::Mat CameraCalibration::processFrame(
        const cv::Mat& raw_frame
    ) {
        
        if (raw_frame.empty()) {
            throw std::invalid_argument(
                "Received empty frame in CameraCalibrator"
            );
        }

        cv::Mat undistorted_frame;
        cv::Mat final_frame;

        // 1. Undistortion
        cv::undistort(
            raw_frame, 
            undistorted_frame, 
            K_inference_, 
            dist_coeffs_
        );

        // 2. Apply pre-computed homography
        cv::warpPerspective(
            undistorted_frame,
            final_frame,
            H_warp_,
            target_size_,
            cv::INTER_LINEAR,
            cv::BORDER_REPLICATE
        );

        return final_frame;

    }

    cv::Mat CameraCalibration::getRotationMatrix(
        double pitch, 
        double yaw, 
        double roll
    ) {
        
        cv::Mat rot_vec = (
            cv::Mat_<double>(3, 1) << pitch, yaw, roll
        );
        cv::Mat R;
        cv::Rodrigues(rot_vec, R);

        return R;

    }


}   // namespace autoware_pov::vision::camera