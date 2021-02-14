#pragma once


#include <opencv2/core.hpp>

#include "Settings.h"

enum class CalibrationState { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

enum class CalibrationResult {FAILED = 0, SUCCESS = 1, WORSE = 2};

CalibrationResult runCalibrationAndSave(const Settings& s, const cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	cv::Mat& rvec, cv::Mat& tvec,
	const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Point3f>& corners, const float grid_width, const bool release_object,
	double& rms, double rmsPrev = std::numeric_limits<double>::infinity(), const bool saveIgnoreRms = true);

void calcBoardCornerPositions(const cv::Size boardSize, const float squareSize, std::vector<cv::Point3f>& corners,
	const Settings::Pattern patternType = Settings::Pattern::CHESSBOARD);