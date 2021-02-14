#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>

class Camera
{
public:

	Camera() : aMatrix(cv::Mat::zeros(3, 3, CV_64FC1)), rMatrix(cv::Mat::zeros(3, 3, CV_64FC1)),
		tMatrix(cv::Mat::zeros(3, 1, CV_64FC1)), pMatrix(cv::Mat::zeros(3, 4, CV_64FC1)), useExtrinsicGuess(false) { }

	void estimatePose(const std::vector<cv::Point3f>& list_points3d,        // list with model 3D coordinates
		const std::vector<cv::Point2f>& list_points2d,        // list with scene 2D coordinates
		cv::Mat& inliers // irnliers container
	);

	inline void setIntrinsics(const cv::Mat& cameraMatrix, const cv::Mat& coeffs)
	{
		cameraMatrix.copyTo(aMatrix);
		coeffs.copyTo(distCoeffs);
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
	}

	inline void setExtrinsics(const cv::Mat& rvec, const cv::Mat& tvec)
	{
		setRot(rvec);
		setTrans(tvec);
		preparePMat();
	}


	inline const cv::Mat &CameraMatrix() const
	{
		return aMatrix;
	}

	inline const cv::Mat& DistCoeffs() const
	{
		return distCoeffs;
	}


private:
	cv::Mat aMatrix;
	cv::Mat rMatrix;
	cv::Mat tMatrix;
	cv::Mat pMatrix;

	cv::Mat rvec_;
	cv::Mat distCoeffs;

	bool useExtrinsicGuess;

	// RANSAC parameters
	int iterationsCount = 500;        // number of Ransac iterations.
	float reprojectionError = 2.0f;    // maximum allowed distance to consider it an inlier.
	float confidence = 0.95f;          // RANSAC successful confidence.
	cv::SolvePnPMethod method = cv::SOLVEPNP_ITERATIVE;


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

};