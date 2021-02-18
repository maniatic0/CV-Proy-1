#include "Camera.h"

#include "Utils.h"

#include <iostream>

void Camera::estimatePose(const std::vector<cv::Point3f>& list_points3d,
	const std::vector<cv::Point2f>& list_points2d,
	const double dt,
	cv::Mat& inliers_idx
)
{
	// Based on https ://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html

	cv::Mat rvec; // output rotation vector
	cv::Mat tvec; // output translation vector

	const bool finalUseExtrinsics = !neverUseExtrinsicGuess && useExtrinsicGuess;
	if (finalUseExtrinsics)
	{
		// initial approximations of the rotation and translation vectors
		rvec_.copyTo(rvec);
		tMatrix.copyTo(tvec);
	}
	else
	{
		// Nothing if we don't use them
		rvec = cv::Mat::zeros(3, 1, CV_64FC1);
		tvec = cv::Mat::zeros(3, 1, CV_64FC1);
	}

	// Use solver to get extrinsic parameters
	const bool res = cv::solvePnPRansac(list_points3d, list_points2d, kMatrix, distCoeffs, rvec, tvec,
		finalUseExtrinsics, iterationsCount, reprojectionError, confidence, inliers_idx, method);

	if (res)
	{
		// Valid solution
		setExtrinsics(rvec, tvec);
	}
	else
	{
		// We failed
		std::cout << "PnP Failed" << std::endl;
		return;
	}

	if (useKalmanFilter)
	{
		// We use filter to improve movement
		// We have to check if this is a good measurment to add to the filter
		const bool goodKalmanMeasurment = inliers_idx.rows >= minInliersKalman;
		if (goodKalmanMeasurment)
		{
			// fill the measurements vector
			fillKalmanMeasurements(measurements, tMatrix, rMatrix);
		}
		else
		{
			std::cout << "Kalman Bad Measurement" << std::endl;
		}

		// update the Kalman filter with good measurements
		updateKalmanFilter(kalmanFilter, measurements, dt, tMatrixKalman, rMatrixKalman);

		if (goodKalmanMeasurment)
		{
			// Update Matrices if we had good information to add from the filter
			tMatrixKalman.copyTo(tMatrix);
			rMatrixKalman.copyTo(rMatrix);
			cv::Rodrigues(rMatrix, rvec_);
			preparePMat();
		}
	}

}

void Camera::updateKalmanFilterDt(cv::KalmanFilter& kalmanFilter, double dt)
{
	// Based on https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	// position
	kalmanFilter.transitionMatrix.at<double>(0, 3) = dt;
	kalmanFilter.transitionMatrix.at<double>(1, 4) = dt;
	kalmanFilter.transitionMatrix.at<double>(2, 5) = dt;
	kalmanFilter.transitionMatrix.at<double>(3, 6) = dt;
	kalmanFilter.transitionMatrix.at<double>(4, 7) = dt;
	kalmanFilter.transitionMatrix.at<double>(5, 8) = dt;
	kalmanFilter.transitionMatrix.at<double>(0, 6) = 0.5 * dt * dt;
	kalmanFilter.transitionMatrix.at<double>(1, 7) = 0.5 * dt * dt;
	kalmanFilter.transitionMatrix.at<double>(2, 8) = 0.5 * dt * dt;
	// orientation
	kalmanFilter.transitionMatrix.at<double>(9, 12) = dt;
	kalmanFilter.transitionMatrix.at<double>(10, 13) = dt;
	kalmanFilter.transitionMatrix.at<double>(11, 14) = dt;
	kalmanFilter.transitionMatrix.at<double>(12, 15) = dt;
	kalmanFilter.transitionMatrix.at<double>(13, 16) = dt;
	kalmanFilter.transitionMatrix.at<double>(14, 17) = dt;
	kalmanFilter.transitionMatrix.at<double>(9, 15) = 0.5 * dt * dt;
	kalmanFilter.transitionMatrix.at<double>(10, 16) = 0.5 * dt * dt;
	kalmanFilter.transitionMatrix.at<double>(11, 17) = 0.5 * dt * dt;

	/* Kalman Model from tutorial*/
	//  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
}

void Camera::resetKalmanFilter(cv::KalmanFilter& kalmanFilter, int nStates, int nMeasurements, int nInputs, double dt)
{
	// Based on https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	// Kalman Filter init
	kalmanFilter.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
	cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
	cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
	cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));             // error covariance

	// Update dt
	updateKalmanFilterDt(kalmanFilter, dt);

	/* Measurment Model from Tutorial */
	//  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

	// Translation Vector
	kalmanFilter.measurementMatrix.at<double>(0, 0) = 1;  // x
	kalmanFilter.measurementMatrix.at<double>(1, 1) = 1;  // y
	kalmanFilter.measurementMatrix.at<double>(2, 2) = 1;  // z

	// Euler angles
	kalmanFilter.measurementMatrix.at<double>(3, 9) = 1;  // roll
	kalmanFilter.measurementMatrix.at<double>(4, 10) = 1; // pitch
	kalmanFilter.measurementMatrix.at<double>(5, 11) = 1; // yaw
}


void Camera::fillKalmanMeasurements(cv::Mat& measurements,
	const cv::Mat& measuredTranslation, const cv::Mat& measuredRotation)
{
	// Based on https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	// Convert rotation matrix to euler angles
	cv::Mat measuredEulers(3, 1, CV_64F);
	measuredEulers = rot2euler(measuredRotation);
	// Set measurement to predict

	// Translation Vector
	measurements.at<double>(0) = measuredTranslation.at<double>(0); // x
	measurements.at<double>(1) = measuredTranslation.at<double>(1); // y
	measurements.at<double>(2) = measuredTranslation.at<double>(2); // z

	// Euler angles
	measurements.at<double>(3) = measuredEulers.at<double>(0);      // roll
	measurements.at<double>(4) = measuredEulers.at<double>(1);      // pitch
	measurements.at<double>(5) = measuredEulers.at<double>(2);      // yaw
}

void Camera::updateKalmanFilter(cv::KalmanFilter& kalmanFilter, cv::Mat& measurement, double dt,
	cv::Mat& estimatedTranslation, cv::Mat& estimatedRotation)
{
	// Based on https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	// Update delta time of model
	updateKalmanFilterDt(kalmanFilter, dt);

	// Update internal model to current time (discard prediction)
	kalmanFilter.predict();
	// The improve measurements by using the filter
	cv::Mat estimated = kalmanFilter.correct(measurement);
	// Estimated translation
	estimatedTranslation.at<double>(0) = estimated.at<double>(0);
	estimatedTranslation.at<double>(1) = estimated.at<double>(1);
	estimatedTranslation.at<double>(2) = estimated.at<double>(2);
	// Estimated euler angles
	cv::Mat estimatedEulers(3, 1, CV_64F);
	estimatedEulers.at<double>(0) = estimated.at<double>(9);
	estimatedEulers.at<double>(1) = estimated.at<double>(10);
	estimatedEulers.at<double>(2) = estimated.at<double>(11);
	// Convert estimated euler angles to rotation matrix
	estimatedRotation = euler2rot(estimatedEulers);
}