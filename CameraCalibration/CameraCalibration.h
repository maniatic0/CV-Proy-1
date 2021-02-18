#pragma once


#include <opencv2/core.hpp>

#include "Settings.h"

/// <summary>
/// Calibration State of the Camera
/// </summary>
enum class CalibrationState { Detection = 0, Capturing = 1, Calibrated = 2 };

/// <summary>
/// Result of the Calibration
/// </summary>
enum class CalibrationResult { Failed = 0, Success = 1, Worse = 2 };

/// <summary>
/// Run a calibration run and save if it is considered better by RMSE
/// </summary>
/// <param name="s">Settings</param>
/// <param name="imageSize">Size of the Image from Camera</param>
/// <param name="cameraMatrix">Output Camera's Intrinsic Matrix</param>
/// <param name="distCoeffs">Output Camera's Distortion Coefficients</param>
/// <param name="rvec">Output Camera's Rotation Coefficients of the last image of the calibration</param>
/// <param name="tvec">Output Camera's Translation Coefficients of the last image of the calibration</param>
/// <param name="imagePoints">Images to use for the calibration</param>
/// <param name="corners">Corners of the board</param>
/// <param name="grid_width">The width of the grid (it is optional to improve the calibration)</param>
/// <param name="release_object">If grid_width is set, this setting turn of a better calibration solver</param>
/// <param name="rms">RMSE of the calibration</param>
/// <param name="rmsPrev">RMSE of the previous calibration (used to reject worse calibrations)</param>
/// <param name="saveIgnoreRms">If the RMSE rejection of worse calibrations is to be ignored</param>
/// <returns>Result of the Calbiration</returns>
CalibrationResult calibrateAndSave(const Settings& s, const cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
	cv::Mat& rvec, cv::Mat& tvec,
	const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Point3f>& corners, const float grid_width, const bool release_object,
	double& rms, double rmsPrev = std::numeric_limits<double>::infinity(), const bool saveIgnoreRms = true);

/// <summary>
/// Calculates the corner's position of the calibration image
/// </summary>
/// <param name="boardSize">Size of the calibration board</param>
/// <param name="squareSize">Size of the squares of the calibration</param>
/// <param name="corners">Output corners of the board (3D world space)</param>
void calcBoardCornerPositions(const cv::Size boardSize, const float squareSize, std::vector<cv::Point3f>& corners);