#pragma once

#include "Settings.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

/// <summary>
/// Pose Estimation based on https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
/// </summary>
class Camera
{
public:

	Camera(Settings s) : kMatrix(cv::Mat::zeros(3, 3, CV_64FC1)), rMatrix(cv::Mat::zeros(3, 3, CV_64FC1)),
		tMatrix(cv::Mat::zeros(3, 1, CV_64FC1)), pMatrix(cv::Mat::zeros(3, 4, CV_64FC1)), world2View(cv::Mat::zeros(3, 4, CV_64FC1)), useExtrinsicGuess(false),
		measurements(cv::Mat::zeros(nMeasurements, 1, CV_64FC1)), rMatrixKalman(cv::Mat::zeros(3, 3, CV_64FC1)), tMatrixKalman(cv::Mat::zeros(3, 1, CV_64FC1)),
		useKalmanFilter(s.useKalmanFilter), expectedDtKalman((double)s.delayUpdate * 1e-3 / (double)CLOCKS_PER_SEC), neverUseExtrinsicGuess(s.neverUseExtrinsicGuess)
	{
		ResetKalmanFilter();
	}

	/// <summary>
	/// Estimate the pose of the camera against the board
	/// </summary>
	/// <param name="list_points3d">Board 3d world coordinates</param>
	/// <param name="list_points2d">Board 2d image plane detected coordinates</param>
	/// <param name="dt">Time from last update</param>
	/// <param name="inliers_idx">Index of accepted points for the estimation (PnP solver, usd in the Kalman Filter)</param>
	void estimatePose(const std::vector<cv::Point3f>& list_points3d,
		const std::vector<cv::Point2f>& list_points2d,
		const double dt,
		cv::Mat& inliers_idx
	);

	/// <summary>
	/// Precalculate World to View Matrix
	/// </summary>
	inline void prepareWorld2View()
	{
		world2View = kMatrix * pMatrix;
	}

	/// <summary>
	/// Set intrisic camera information
	/// </summary>
	/// <param name="cameraMatrix">Camera's Instrinsic Matrix</param>
	/// <param name="coeffs">Camera's Distortion Coefficients</param>
	inline void setIntrinsics(const cv::Mat& cameraMatrix, const cv::Mat& coeffs)
	{
		cameraMatrix.copyTo(kMatrix);
		coeffs.copyTo(distCoeffs);
		prepareWorld2View();
		ResetKalmanFilter();
	}

	/// <summary>
	/// Set Camera Rotation
	/// </summary>
	/// <param name="rvec">Rotation Vector</param>
	inline void setRot(const cv::Mat& rvec)
	{
		rvec.copyTo(rvec_);
		cv::Rodrigues(rvec, rMatrix);
	}

	/// <summary>
	/// Set Camera Translation
	/// </summary>
	/// <param name="tvec">Translation Vector</param>
	inline void setTrans(const cv::Mat& tvec)
	{
		tvec.copyTo(tMatrix);
	}

	/// <summary>
	/// If the pose estimation should use the previous rotation and translation for guess
	/// </summary>
	/// <param name="useGuess"></param>
	inline void setUseExtrinsicGuess(const bool useGuess)
	{
		useExtrinsicGuess = useGuess;
	}

	/// <summary>
	/// Prepare all the matrices and precalculations
	/// </summary>
	inline void preparePMat()
	{
		set_P_matrix(rMatrix, tMatrix);
		setUseExtrinsicGuess(true);
		prepareWorld2View();
	}

	/// <summary>
	/// Set Camera's extrinsic information
	/// </summary>
	/// <param name="rvec">Camera's Rotation Vector</param>
	/// <param name="tvec">Camera's Translation Vector</param>
	inline void setExtrinsics(const cv::Mat& rvec, const cv::Mat& tvec)
	{
		setRot(rvec);
		setTrans(tvec);
		preparePMat();
	}

	/// <summary>
	/// Get Camera's Instrinsic Matrix
	/// </summary>
	/// <returns></returns>
	inline const cv::Mat& CameraMatrix() const
	{
		return kMatrix;
	}

	/// <summary>
	/// Get Distortion Coefficients
	/// </summary>
	/// <returns></returns>
	inline const cv::Mat& DistCoeffs() const
	{
		return distCoeffs;
	}

	/// <summary>
	/// Rotation Vector
	/// </summary>
	/// <returns></returns>
	inline const cv::Mat& RotationVec() const
	{
		return rvec_;
	}

	/// <summary>
	/// Translation Vector
	/// </summary>
	/// <returns></returns>
	inline const cv::Mat& TranslationVec() const
	{
		return tMatrix;
	}

	/// <summary>
	/// Project a 3D Point in world space to 2D point in colorBuff space with Depth
	/// </summary>
	/// <param name="point">3D Point to Project</param>
	/// <returns>2D Point with depthBuff</returns>
	inline cv::Point3f projectPoint(const cv::Point3f& point)
	{

		cv::Mat point3dHomogenous = cv::Mat(4, 1, CV_64FC1);
		point3dHomogenous.at<double>(0) = point.x;
		point3dHomogenous.at<double>(1) = point.y;
		point3dHomogenous.at<double>(2) = -point.z; // Fix for left hand side
		point3dHomogenous.at<double>(3) = 1;
		// 2D point		vector [u v 1]'
		cv::Mat pointProj = cv::Mat(4, 1, CV_64FC1);
		pointProj = world2View * point3dHomogenous;

		// Perspective projection final step
		cv::Point3f point2d;
		point2d.x = (float)(pointProj.at<double>(0) / pointProj.at<double>(2));
		point2d.y = (float)(pointProj.at<double>(1) / pointProj.at<double>(2));
		point2d.z = (float)pointProj.at<double>(2);
		return point2d;
	}


	/// <summary>
	/// Project a 3D Point in world space to 2D point in colorBuff space
	/// </summary>
	/// <param name="point">3D Point to Project</param>
	/// <returns>2D Point</returns>
	inline cv::Point2f projectPointNoDepth(const cv::Point3f& point)
	{
		cv::Point3f p = projectPoint(point);
		return cv::Point2f(p.x, p.y);
	}

private:
	cv::Mat kMatrix;
	cv::Mat rMatrix;
	cv::Mat tMatrix;
	cv::Mat pMatrix;

	cv::Mat world2View;

	cv::Mat rvec_;
	cv::Mat distCoeffs;

	bool useExtrinsicGuess; // Use extrinsic guess from the last frame
	bool neverUseExtrinsicGuess; // Never use extrinsic guess

	// RANSAC parameters
	int iterationsCount = 500;        // number of Ransac iterations.
	float reprojectionError = 2.0f;    // maximum allowed distance to consider it an inlier.
	float confidence = 0.95f;          // RANSAC successful confidence.
	cv::SolvePnPMethod method = cv::SOLVEPNP_EPNP;

	// Kalman Filter parameters
	cv::KalmanFilter KF;			// instantiate Kalman Filter
	int nStates = 18;				// the number of states
	int nMeasurements = 6;			// the number of measured states
	int nInputs = 0;				// the number of action control
	int minInliersKalman = 30;		// Kalman threshold updatin
	cv::Mat measurements;			// Kalman measurements from the last valid frame
	cv::Mat rMatrixKalman;			// Kalman Rotation Estimation
	cv::Mat tMatrixKalman;			// Kalman Translation Estimation
	double expectedDtKalman;		// Kalman expected dt

	bool useKalmanFilter;

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
		resetKalmanFilter(KF, nStates, nMeasurements, nInputs, expectedDtKalman); // Last one is the estimated dt
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