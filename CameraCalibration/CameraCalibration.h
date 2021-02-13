#pragma once


#include <opencv2/core.hpp>

#include "Settings.h"

enum class CalibrationState { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

bool runCalibrationAndSave(const Settings& s, const cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	const std::vector<std::vector<cv::Point2f> > &imagePoints, const float grid_width, const bool release_object);

void calcBoardCornerPositions(const cv::Size boardSize, const float squareSize, std::vector<cv::Point3f>& corners,
	const Settings::Pattern patternType = Settings::Pattern::CHESSBOARD);