#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

/// <summary>
/// Based On https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
/// </summary>
class Camera
{
public:

	Camera() : aMatrix(cv::Mat::zeros(3, 3, CV_64FC1)), rMatrix(cv::Mat::zeros(3, 3, CV_64FC1)),
		tMatrix(cv::Mat::zeros(3, 1, CV_64FC1)), pMatrix(cv::Mat::zeros(3, 4, CV_64FC1)), world2View(cv::Mat::zeros(3, 4, CV_64FC1)), useExtrinsicGuess(false),
		measurements(cv::Mat::zeros(nMeasurements, 1, CV_64FC1)), rMatrixKalman(cv::Mat::zeros(3, 3, CV_64FC1)), tMatrixKalman(cv::Mat::zeros(3, 1, CV_64FC1))
	{
		ResetKalmanFilter();
	}

	void estimatePose(const std::vector<cv::Point3f>& list_points3d,        // list with model 3D coordinates
		const std::vector<cv::Point2f>& list_points2d,        // list with scene 2D coordinates
		const double dt,
		cv::Mat& inliers_idx // irnliers container
	);

	inline void prepareWorld2View()
	{
		world2View = aMatrix * pMatrix;
	}

	inline void setIntrinsics(const cv::Mat& cameraMatrix, const cv::Mat& coeffs)
	{
		cameraMatrix.copyTo(aMatrix);
		coeffs.copyTo(distCoeffs);
		prepareWorld2View();
		ResetKalmanFilter();
	}

	inline void setRot(const cv::Mat& rvec)
	{
		rvec.copyTo(rvec_);
		cv::Rodrigues(rvec, rMatrix);
	}

	inline void setTrans(const cv::Mat& tvec)
	{
		tvec.copyTo(tMatrix);
	}

	inline void setUseExtrinsicGuess(const bool useGuess)
	{
		useExtrinsicGuess = useGuess;
	}

	inline void preparePMat()
	{
		set_P_matrix(rMatrix, tMatrix);
		setUseExtrinsicGuess(true);
		prepareWorld2View();
	}

	inline void setExtrinsics(const cv::Mat& rvec, const cv::Mat& tvec)
	{
		setRot(rvec);
		setTrans(tvec);
		preparePMat();
	}

	inline const cv::Mat& CameraMatrix() const
	{
		return aMatrix;
	}

	inline const cv::Mat& DistCoeffs() const
	{
		return distCoeffs;
	}

	inline const cv::Mat& RotationVec() const
	{
		return rvec_;
	}

	inline const cv::Mat& TranslationVec() const
	{
		return tMatrix;
	}

	inline cv::Point2f projectPoint(const cv::Point3f& point)
	{
		// 3D point vector [x y z 1]'
		cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
		point3d_vec.at<double>(0) = point.x;
		point3d_vec.at<double>(1) = point.y;
		point3d_vec.at<double>(2) = point.z;
		point3d_vec.at<double>(3) = 1;
		// 2D point		vector [u v 1]'
		cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
		point2d_vec = world2View * point3d_vec;
		// Normalization of [u v]'
		cv::Point2f point2d;
		point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
		point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));
		return point2d;
	}


private:
	cv::Mat aMatrix;
	cv::Mat rMatrix;
	cv::Mat tMatrix;
	cv::Mat pMatrix;

	cv::Mat world2View;

	cv::Mat rvec_;
	cv::Mat distCoeffs;

	bool useExtrinsicGuess;

	// RANSAC parameters
	int iterationsCount = 500;        // number of Ransac iterations.
	float reprojectionError = 2.0f;    // maximum allowed distance to consider it an inlier.
	float confidence = 0.95f;          // RANSAC successful confidence.
	cv::SolvePnPMethod method = cv::SOLVEPNP_ITERATIVE;

	// Kalman Filter parameters
	cv::KalmanFilter KF;			// instantiate Kalman Filter
	int nStates = 18;				// the number of states
	int nMeasurements = 6;			// the number of measured states
	int nInputs = 0;				// the number of action control
	int minInliersKalman = 30;		// Kalman threshold updatin
	cv::Mat measurements;			// Kalman measurements from the last valid frame
	cv::Mat rMatrixKalman;			// Kalman Rotation Estimation
	cv::Mat tMatrixKalman;			// Kalman Translation Estimation


	/// <summary>
	/// Set Projection Matrix
	/// </summary>
	/// <param name="R_matrix">Rotation Matrix</param>
	/// <param name="t_matrix">Translation Matrix</param>
	inline void set_P_matrix(const cv::Mat& R_matrix, const cv::Mat& t_matrix)
	{
		// Rotation-Translation Matrix Definition
		pMatrix.at<double>(0, 0) = R_matrix.at<double>(0, 0);
		pMatrix.at<double>(0, 1) = R_matrix.at<double>(0, 1);
		pMatrix.at<double>(0, 2) = R_matrix.at<double>(0, 2);
		pMatrix.at<double>(1, 0) = R_matrix.at<double>(1, 0);
		pMatrix.at<double>(1, 1) = R_matrix.at<double>(1, 1);
		pMatrix.at<double>(1, 2) = R_matrix.at<double>(1, 2);
		pMatrix.at<double>(2, 0) = R_matrix.at<double>(2, 0);
		pMatrix.at<double>(2, 1) = R_matrix.at<double>(2, 1);
		pMatrix.at<double>(2, 2) = R_matrix.at<double>(2, 2);
		pMatrix.at<double>(0, 3) = t_matrix.at<double>(0);
		pMatrix.at<double>(1, 3) = t_matrix.at<double>(1);
		pMatrix.at<double>(2, 3) = t_matrix.at<double>(2);
	}

	/// <summary>
	/// Update delta time for Kalman Filter
	/// </summary>
	/// <param name="KF">Kalman Filter to update</param>
	/// <param name="dt">Delta time</param>
	static void updateKalmanFilterDt(cv::KalmanFilter& KF, double dt);

	/// <summary>
	/// Resets a Kalman Filter to start learning again
	/// </summary>
	/// <param name="KF">Kalman Fiter</param>
	/// <param name="nStates">Number of States for the Kalman Filter</param>
	/// <param name="nMeasurements">Number of Measurements for the Kalman Filter</param>
	/// <param name="nInputs"></param>
	/// <param name="dt"></param>
	static void resetKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt);

	/// <summary>
	/// Resets Kalman Filter
	/// </summary>
	inline void ResetKalmanFilter()
	{
		resetKalmanFilter(KF, nStates, nMeasurements, nInputs, 0.125); // Last one is the estimated dt
	}

	/// <summary>
	/// Fill Kalman Measurements (translation + angles)
	/// </summary>
	/// <param name="measurements">Kalman Measurements</param>
	/// <param name="translation_measured">Translation Measured</param>
	/// <param name="rotation_measured">Rotation Measured</param>
	static void fillKalmanMeasurements(cv::Mat& measurements,
		const cv::Mat& translation_measured, const cv::Mat& rotation_measured);

	/// <summary>
	/// Update Kalman Filter with current measurements
	/// </summary>
	/// <param name="KF">Kalman Filter</param>
	/// <param name="measurement">New Measurements</param>
	/// <param name="translation_estimated">Estimated Translation</param>
	/// <param name="rotation_estimated">Estimated Rotation</param>
	static void updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurement, double dt,
		cv::Mat& translation_estimated, cv::Mat& rotation_estimated);

};