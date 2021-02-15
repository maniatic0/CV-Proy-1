#include "CameraCalibration.h"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "Settings.h"

//! [compute_errors]
static double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints,
	const std::vector<std::vector<cv::Point2f> >& imagePoints,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
	std::vector<float>& perViewErrors, const bool fisheye)
{
	std::vector<cv::Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		if (fisheye)
		{
			cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
				distCoeffs);
		}
		else
		{
			cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		}
		err = cv::norm(imagePoints[i], imagePoints2, cv::NORM_L2);

		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}
//! [compute_errors]

//! [board_corners]
void calcBoardCornerPositions(const cv::Size boardSize, const float squareSize, std::vector<cv::Point3f>& corners,
	const Settings::Pattern patternType)
{
	corners.clear();

	switch (patternType)
	{
	case Settings::Pattern::CHESSBOARD:
	case Settings::Pattern::CIRCLES_GRID:
	{
		for (int i = 0; i < boardSize.height; ++i)
		{
			for (int j = 0; j < boardSize.width; ++j)
			{
				corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
			}
		}
	}
	break;

	case Settings::Pattern::ASYMMETRIC_CIRCLES_GRID:
	{
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				corners.push_back(cv::Point3f((2 * j + i % 2) * squareSize, i * squareSize, 0));
			}
		}

	}
	break;
	default:
		break;
	}
}
//! [board_corners]

static bool runCalibration(const Settings& s, const cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Point3f>& corners,
	std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
	std::vector<float>& reprojErrs, double& totalAvgErr, double& rms, std::vector<cv::Point3f>& newObjPoints,
	float grid_width, bool release_object)
{
	//! [fixed_aspect]
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	if (s.flag & cv::CALIB_FIX_ASPECT_RATIO)
	{
		cameraMatrix.at<double>(0, 0) = s.aspectRatio;
	}
	//! [fixed_aspect]
	if (s.useFisheye) {
		distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	}
	else {
		distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
	}

	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	objectPoints[0] = corners;
	objectPoints[0][(size_t)(s.boardSize.width) - 1].x = objectPoints[0][0].x + grid_width;
	newObjPoints = objectPoints[0];

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters

	if (s.useFisheye) {
		cv::Mat _rvecs, _tvecs;
		rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
			_tvecs, s.flag);

		rvecs.reserve(_rvecs.rows);
		tvecs.reserve(_tvecs.rows);
		for (int i = 0; i < int(objectPoints.size()); i++) {
			rvecs.push_back(_rvecs.row(i));
			tvecs.push_back(_tvecs.row(i));
		}
	}
	else {
		int iFixedPoint = -1;
		if (release_object)
		{
			iFixedPoint = s.boardSize.width - 1;
		}
		rms = cv::calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
			cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
			s.flag | cv::CALIB_USE_LU);
	}

	if (release_object) {
		std::cout << "New board corners: " << std::endl;
		std::cout << newObjPoints[0] << std::endl;
		std::cout << newObjPoints[(size_t)(s.boardSize.width) - 1] << std::endl;
		std::cout << newObjPoints[(size_t)(s.boardSize.width) * ((size_t)(s.boardSize.height) - 1)] << std::endl;
		std::cout << newObjPoints.back() << std::endl;
	}

	std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	objectPoints.clear();
	objectPoints.resize(imagePoints.size(), newObjPoints);
	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
		distCoeffs, reprojErrs, s.useFisheye);

	return ok;
}

// Print camera parameters to the output file
static void saveCameraParams(const Settings& s, const cv::Size& imageSize, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const std::vector<float>& reprojErrs, const std::vector<std::vector<cv::Point2f> >& imagePoints,
	const double totalAvgErr, const std::vector<cv::Point3f>& newObjPoints)
{
	cv::FileStorage fs(s.outputFileName, cv::FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm t2;
	localtime_s(&t2, &tm);
	char buf[1024];
	strftime(buf, sizeof(buf), "%c", &t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
	{
		fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
	}
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << s.boardSize.width;
	fs << "board_height" << s.boardSize.height;
	fs << "square_size" << s.squareSize;

	if (s.flag & cv::CALIB_FIX_ASPECT_RATIO)
	{
		fs << "fix_aspect_ratio" << s.aspectRatio;
	}

	if (s.flag)
	{
		std::stringstream flagsStringStream;
		if (s.useFisheye)
		{
			flagsStringStream << "flags:"
				<< (s.flag & cv::fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
				<< (s.flag & cv::fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
				<< (s.flag & cv::fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
				<< (s.flag & cv::fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
				<< (s.flag & cv::fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
				<< (s.flag & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
		}
		else
		{
			flagsStringStream << "flags:"
				<< (s.flag & cv::CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
				<< (s.flag & cv::CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
				<< (s.flag & cv::CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
				<< (s.flag & cv::CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
				<< (s.flag & cv::CALIB_FIX_K1 ? " +fix_k1" : "")
				<< (s.flag & cv::CALIB_FIX_K2 ? " +fix_k2" : "")
				<< (s.flag & cv::CALIB_FIX_K3 ? " +fix_k3" : "")
				<< (s.flag & cv::CALIB_FIX_K4 ? " +fix_k4" : "")
				<< (s.flag & cv::CALIB_FIX_K5 ? " +fix_k5" : "");
		}
		fs.writeComment(flagsStringStream.str());
	}

	fs << "flags" << s.flag;

	fs << "fisheye_model" << s.useFisheye;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (s.writeExtrinsics && !reprojErrs.empty())
		fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

	if (s.writeExtrinsics && !rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		cv::Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
		bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
		bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

		for (size_t i = 0; i < rvecs.size(); i++)
		{
			cv::Mat r = bigmat(cv::Range(int(i), int(i + 1)), cv::Range(0, 3));
			cv::Mat t = bigmat(cv::Range(int(i), int(i + 1)), cv::Range(3, 6));

			if (needReshapeR)
			{
				rvecs[i].reshape(1, 1).copyTo(r);
			}
			else
			{
				//*.t() is MatExpr (not Mat) so we can use assignment operator
				CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
				r = rvecs[i].t();
			}

			if (needReshapeT)
			{
				tvecs[i].reshape(1, 1).copyTo(t);
			}
			else
			{
				CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
				t = tvecs[i].t();
			}
		}
		fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
		fs << "extrinsic_parameters" << bigmat;
	}

	if (s.writePoints && !imagePoints.empty())
	{
		cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (size_t i = 0; i < imagePoints.size(); i++)
		{
			cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
			cv::Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}

	if (s.writeGrid && !newObjPoints.empty())
	{
		fs << "grid_points" << newObjPoints;
	}
}

//! [run_and_save]
CalibrationResult runCalibrationAndSave(const Settings& s, const cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	cv::Mat& rvec, cv::Mat& tvec,
	const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Point3f>& corners, const float grid_width, const bool release_object,
	double& rms, double rmsPrev, const bool saveIgnoreRms)
{
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	std::vector<float> reprojErrs;
	double totalAvgErr = 0;
	std::vector<cv::Point3f> newObjPoints;

	bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, corners, rvecs, tvecs, reprojErrs,
		totalAvgErr, rms, newObjPoints, grid_width, release_object);
	std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr << std::endl;

	bool betterRMS = rms < rmsPrev;

	if (ok && (saveIgnoreRms || betterRMS))
	{
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
			totalAvgErr, newObjPoints);
	}

	if (!ok)
	{
		std::cout << "Failed to Converge Calibration" << std::endl;
		return CalibrationResult::FAILED;
	}

	if (rvecs.size() > 0)
	{
		rvecs[rvecs.size() - 1].copyTo(rvec);
		tvecs[tvecs.size() - 1].copyTo(tvec);
	}	

	if (betterRMS)
	{
		std::cout << "Converged to a better Calibration: " << rms << " < " << rmsPrev << std::endl;
		return CalibrationResult::SUCCESS;
	}
	else if (saveIgnoreRms)
	{
		std::cout << "Converged to a worse Calibration but we forced to accept it: " << rms << " >= " << rmsPrev << std::endl;
		return CalibrationResult::SUCCESS;
	}

	std::cout << "Converged to a worse Calibration: " << rms << " >= " << rmsPrev << std::endl;
	return CalibrationResult::WORSE;
}
//! [run_and_save]