#include "Camera.h"

void Camera::estimatePose(const std::vector<cv::Point3f>& list_points3d,        // list with model 3D coordinates
	const std::vector<cv::Point2f>& list_points2d,        // list with scene 2D coordinates
	cv::Mat& inliers // irnliers container
) 
{
    cv::Mat rvec; // output rotation vector
    cv::Mat tvec; // output translation vector
    // initial approximations of the rotation and translation vectors
           

    if (useExtrinsicGuess)
    {
        rvec_.copyTo(rvec);
        tMatrix.copyTo(tvec);
    }
    else
    {
        rvec = cv::Mat::zeros(3, 1, CV_64FC1);
        tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    }
                                      
    const bool res = cv::solvePnPRansac(list_points3d, list_points2d, aMatrix, distCoeffs, rvec, tvec,
        useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, method);
    if (res)
    {
        setExtrinsics(rvec, tvec);
    }
}