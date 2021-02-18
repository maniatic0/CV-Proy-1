#include "Camera.h"

#include "Utils.h"

#include <iostream>

void Camera::estimatePose(const std::vector<cv::Point3f>& list_points3d,        // list with model 3D coordinates
	const std::vector<cv::Point2f>& list_points2d,        // list with scene 2D coordinates
	const double dt,
	cv::Mat& inliers_idx // irnliers id container (from PnP)
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

	const bool res = cv::solvePnPRansac(list_points3d, list_points2d, kMatrix, distCoeffs, rvec, tvec,
		useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers_idx, method);

	if (res)
	{
		setExtrinsics(rvec, tvec);
	}
	else
	{
		std::cout << "PnP Failed" << std::endl;
	}

	if (useKalmanFilter)
	{
		// GOOD MEASUREMENT
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
		updateKalmanFilter(KF, measurements, dt, tMatrixKalman, rMatrixKalman);

		if (goodKalmanMeasurment)
		{
			// Update Matrices
			tMatrixKalman.copyTo(tMatrix);
			rMatrixKalman.copyTo(rMatrix);
			cv::Rodrigues(rMatrix, rvec_);
			preparePMat();
		}
	}

}

void Camera::updateKalmanFilterDt(cv::KalmanFilter& KF, double dt)
{
	// position
	KF.transitionMatrix.at<double>(0, 3) = dt;
	KF.transitionMatrix.at<double>(1, 4) = dt;
	KF.transitionMatrix.at<double>(2, 5) = dt;
	KF.transitionMatrix.at<double>(3, 6) = dt;
	KF.transitionMatrix.at<double>(4, 7) = dt;
	KF.transitionMatrix.at<double>(5, 8) = dt;
	KF.transitionMatrix.at<double>(0, 6) = 0.5 * dt * dt;
	KF.transitionMatrix.at<double>(1, 7) = 0.5 * dt * dt;
	KF.transitionMatrix.at<double>(2, 8) = 0.5 * dt * dt;
	// orientation
	KF.transitionMatrix.at<double>(9, 12) = dt;
	KF.transitionMatrix.at<double>(10, 13) = dt;
	KF.transitionMatrix.at<double>(11, 14) = dt;
	KF.transitionMatrix.at<double>(12, 15) = dt;
	KF.transitionMatrix.at<double>(13, 16) = dt;
	KF.transitionMatrix.at<double>(14, 17) = dt;
	KF.transitionMatrix.at<double>(9, 15) = 0.5 * dt * dt;
	KF.transitionMatrix.at<double>(10, 16) = 0.5 * dt * dt;
	KF.transitionMatrix.at<double>(11, 17) = 0.5 * dt * dt;
}

void Camera::resetKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt)
{
	KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
	cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
	cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
				   /* DYNAMIC MODEL */
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
	updateKalmanFilterDt(KF, dt);
	/* MEASUREMENT MODEL */
	//  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
	KF.measurementMatrix.at<double>(0, 0) = 1;  // x
	KF.measurementMatrix.at<double>(1, 1) = 1;  // y
	KF.measurementMatrix.at<double>(2, 2) = 1;  // z
	KF.measurementMatrix.at<double>(3, 9) = 1;  // roll
	KF.measurementMatrix.at<double>(4, 10) = 1; // pitch
	KF.measurementMatrix.at<double>(5, 11) = 1; // yaw
}


void Camera::fillKalmanMeasurements(cv::Mat& measurements,
	const cv::Mat& translation_measured, const cv::Mat& rotation_measured)
{
	// Convert rotation matrix to euler angles
	cv::Mat measured_eulers(3, 1, CV_64F);
	measured_eulers = rot2euler(rotation_measured);
	// Set measurement to predict
	measurements.at<double>(0) = translation_measured.at<double>(0); // x
	measurements.at<double>(1) = translation_measured.at<double>(1); // y
	measurements.at<double>(2) = translation_measured.at<double>(2); // z
	measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
	measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
	measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

void Camera::updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurement, double dt,
	cv::Mat& translation_estimated, cv::Mat& rotation_estimated)
{
	// Update delta time
	updateKalmanFilterDt(KF, dt);

	// First predict, to update the internal statePre variable
	cv::Mat prediction = KF.predict();
	// The "correct" phase that is going to use the predicted value and our measurement
	cv::Mat estimated = KF.correct(measurement);
	// Estimated translation
	translation_estimated.at<double>(0) = estimated.at<double>(0);
	translation_estimated.at<double>(1) = estimated.at<double>(1);
	translation_estimated.at<double>(2) = estimated.at<double>(2);
	// Estimated euler angles
	cv::Mat eulers_estimated(3, 1, CV_64F);
	eulers_estimated.at<double>(0) = estimated.at<double>(9);
	eulers_estimated.at<double>(1) = estimated.at<double>(10);
	eulers_estimated.at<double>(2) = estimated.at<double>(11);
	// Convert estimated quaternion to rotation matrix
	rotation_estimated = euler2rot(eulers_estimated);
}