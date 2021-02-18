#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "CameraCalibration.h"
#include "Settings.h"
#include "Camera.h"
#include "ZBuffer.h"

// Based on https://docs.opencv.org/4.2.0/d4/d94/tutorial_camera_calibration.html

int main(int argc, char* argv[])
{
	const cv::String keys
		= "{help h usage ? |           | print this message            }"
		"{@settings      |default.xml| input setting file            }";
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("This is a camera calibration sample.\n"
		"Usage: camera_calibration [configuration_file -- default ./default.xml]\n"
		"Near the sample file you'll find the configuration file, which has detailed help of "
		"how to edit it. It may be any OpenCV supported file format XML/YAML.");
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	Settings s;
	{
		const std::string inputSettingsFile = parser.get<std::string>(0);
		cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
		if (!fs.isOpened())
		{
			std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
			parser.printMessage();
			return -1;
		}
		if (!Settings::readFromFile(fs, s))
		{
			fs.release(); // close Settings file
			std::cout << "Invalid input detected. Application stopping. " << std::endl;
			return -1;
		}
		fs.release(); // close Settings file
	}

	int winSize = (int)(s.squareSize / 2.0f);

	float grid_width = s.gridWidth;
	bool release_object = s.releaseObject;

	// Only do this once
	std::vector<cv::Point3f> corners;
	calcBoardCornerPositions(s.boardSize, s.squareSize, corners);

	float grid_height = s.squareSize * (float)(s.boardSize.height - 1);

	// Render stuff
	Camera camera(s);
	cv::Mat inliers;
	std::vector<std::vector<cv::Point2f> > imagePoints;
	cv::Mat cameraMatrixTemp, distCoeffsTemp;
	cv::Mat rvecTemp, tvecTemp;
	double rms = std::numeric_limits<double>::infinity();
	double rmsTemp = std::numeric_limits<double>::infinity();
	int currRestarts = s.restartAttemps;
	int currAttempts = s.nrFrames;
	bool atLeastOneSuccesss = false;
	size_t preImagePointsSize = imagePoints.size();
	cv::Size imageSize;
	CalibrationState mode = s.inputType == Settings::InputType::Image_List ? CalibrationState::Capturing : CalibrationState::Detection;
	clock_t prevTimestamp = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> prevFrame = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> startAnimTime = std::chrono::high_resolution_clock::now();
	double dt;
	double animTime = 0;
	const cv::Scalar RED(0, 0, 255); // BGR
	const cv::Scalar GREEN(0, 255, 0); // BGR
	const cv::Scalar BLUE(255, 0, 0); // BGR

	constexpr char ESC_KEY = 27;

	// For Axis Purposes
	cv::Point3f origin(0, 0, 0);
	int axisThickness = 3;
	float axisMultiplier = 4.0f;
	cv::Point3f xAxis(s.squareSize * axisMultiplier, 0, 0);
	cv::Point3f yAxis(0, s.squareSize * axisMultiplier, 0);
	cv::Point3f zAxis(0, 0, s.squareSize * axisMultiplier);
	float axisTextOffset = 2.0f;
	cv::Point3f xAxisText(s.squareSize * axisMultiplier + axisTextOffset, 0, 0);
	cv::Point3f yAxisText(0, s.squareSize * axisMultiplier + axisTextOffset, 0);
	cv::Point3f zAxisText(0, 0, s.squareSize * axisMultiplier + axisTextOffset);

	// For Cube
	std::vector<cv::Point3f> points =
	{
		cv::Point3f(0, 0, 0),
		cv::Point3f(0, s.squareSize * 2.0f, 0),
		cv::Point3f(0, s.squareSize * 2.0f, s.squareSize * 2.0f),
		cv::Point3f(0, 0, s.squareSize * 2.0f),
		cv::Point3f(s.squareSize * 2.0f, 0, 0),
		cv::Point3f(s.squareSize * 2.0f, s.squareSize * 2.0f, 0),
		cv::Point3f(s.squareSize * 2.0f, s.squareSize * 2.0f, s.squareSize * 2.0f),
		cv::Point3f(s.squareSize * 2.0f, 0, s.squareSize * 2.0f)
	};
	const size_t cubeLinesNumber = 12;
	constexpr int cubeLines[cubeLinesNumber][2] =
	{
		{0, 1},
		{1, 2},
		{2, 3},
		{3, 0},
		{0, 4},
		{1, 5},
		{2, 6},
		{3, 7},
		{4, 5},
		{5, 6},
		{6, 7},
		{7, 4}
	};
	const cv::Scalar cubeColor(0, 255, 255);
	int cubeThickness = 7;

	// Poor man Z Buffering. Everything is a transparency
	ZBuffer zBuffer;

	while (true)
	{
		dt = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - prevFrame).count();


		bool blinkOutput = false;

		zBuffer.setColor(s.nextImage());
		cv::Mat& view = zBuffer.getColor();

		//If we are capturing to calibrate and we have enough images to attempt to calibrate and we accepted a new image to try to calibrate
		if (mode == CalibrationState::Capturing && imagePoints.size() >= 1 && preImagePointsSize < imagePoints.size())
		{
			--currAttempts;
			const CalibrationResult res = calibrateAndSave(s, imageSize, cameraMatrixTemp, distCoeffsTemp, rvecTemp, tvecTemp, imagePoints, corners, grid_width,
				release_object, rmsTemp, rms, !atLeastOneSuccesss);
			if (res != CalibrationResult::Failed)
			{
				atLeastOneSuccesss = true;
				if (res == CalibrationResult::Success)
				{
					// We are better
					rms = rmsTemp;
					camera.setIntrinsics(cameraMatrixTemp, distCoeffsTemp);
					camera.setExtrinsics(rvecTemp, tvecTemp);


					if (imagePoints.size() >= (size_t)s.nrFrames)
					{
						// We are done by Number of Frames
						std::cout << "We are calibrated by Number of Frames!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
					}
					else if (rms <= s.acceptableThreshold)
					{
						// We are done by Threshold
						std::cout << "We are calibrated by Threshold!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
					}
					else
					{
						std::cout << "We improve the calibration!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
					}
				}
				else
				{
					// We are worse

					// Pop last image
					imagePoints.pop_back();
					std::cout << "We are not calibrated: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
				}
			}
			else
			{
				// mode = CalibrationState::Detection; // ?
				std::cout << "We failed to calibrate!" << std::endl;
			}

			if (currRestarts >= 0)
			{
				if (currAttempts < 0)
				{
					--currRestarts;
					currAttempts = s.nrFrames;
					imagePoints.clear();
					std::cout << "Restarting Calibration. Current RMS: " << rms << ". Attempt: " << currRestarts << std::endl;
				}
			}
			else
			{
				if (atLeastOneSuccesss)
				{
					mode = CalibrationState::Calibrated;
					std::cout << "We settle Calibration with RMS: " << rms << std::endl;
				}
				else
				{
					mode = s.inputType == Settings::InputType::Image_List ? CalibrationState::Capturing : CalibrationState::Detection;
					std::cout << "We failed to calibrate completly!" << std::endl;
				}
			}
		}
		preImagePointsSize = imagePoints.size();
		if (view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if (!imagePoints.empty())
			{
				const CalibrationResult res = calibrateAndSave(s, imageSize, cameraMatrixTemp, distCoeffsTemp, rvecTemp, tvecTemp, imagePoints, corners, grid_width,
					release_object, rmsTemp, rms, !atLeastOneSuccesss);
				if (res != CalibrationResult::Failed)
				{
					atLeastOneSuccesss = true;
					if (res == CalibrationResult::Success)
					{
						// We are better
						rms = rmsTemp;
						camera.setIntrinsics(cameraMatrixTemp, distCoeffsTemp);
						camera.setExtrinsics(rvecTemp, tvecTemp);

						if (imagePoints.size() >= (size_t)s.nrFrames)
						{
							// We are done
							mode = CalibrationState::Calibrated;
							std::cout << "We are calibrated!" << std::endl;
						}
						else
						{
							std::cout << "We improved the calibration!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
						}
					}
					else
					{
						// We are worse
						std::cout << "We are not calibrated: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
					}
				}
				else
				{
					std::cout << "We failed to calibrate!" << std::endl;
				}
			}
			break;
		}

		imageSize = view.size();  // Format input image.
		if (s.flipVertical)
		{
			// If for some reason the image is vertically flipped
			flip(view, view, 0);
		}

		// Detected 2D corners
		std::vector<cv::Point2f> pointBuf;

		int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

		if (!s.useFisheye) {
			// fast check erroneously fails with high distortions like fisheye
			chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
		}

		// Find Chessboard
		const bool found = cv::findChessboardCorners(view, s.boardSize, pointBuf, chessBoardFlags);

		if (found)
		{
			// If we found the chessboard

			// chessboard found 2D corners can be improved with a solver. It requires everything in black and white
			cv::Mat viewGray;
			cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, cv::Size(winSize, winSize), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));

			// Draw the corners.
			drawChessboardCorners(view, s.boardSize, cv::Mat(pointBuf), found);

			switch (mode)
			{
			case CalibrationState::Capturing:
			{
				if (!s.inputCapture.isOpened() || clock() - prevTimestamp > (clock_t)s.delay * 1e-3 * CLOCKS_PER_SEC)
				{
					// For camera only take new samples after delay time
					imagePoints.push_back(pointBuf);
					prevTimestamp = clock();
					blinkOutput = s.inputCapture.isOpened();
				}
			}
			break;
			case CalibrationState::Calibrated:
			{
				if (clock() - prevTimestamp > (clock_t)s.delayUpdate * 1e-3 * CLOCKS_PER_SEC)
				{
					// For camera only take new samples after delay time
					camera.estimatePose(corners, pointBuf, (double)(clock() - prevTimestamp) / (double)CLOCKS_PER_SEC, inliers);
					prevTimestamp = clock();
				}

				{
					// Axis Drawing
					cv::Point3f ori = camera.projectPoint(origin);
					cv::Point3f xAxis2d = camera.projectPoint(xAxis);
					cv::Point3f yAxis2d = camera.projectPoint(yAxis);
					cv::Point3f zAxis2d = camera.projectPoint(zAxis);

					zBuffer.drawLine(ori, xAxis2d, RED, axisThickness);
					zBuffer.drawLine(ori, yAxis2d, GREEN, axisThickness);
					zBuffer.drawLine(ori, zAxis2d, BLUE, axisThickness);
				}

				{
					// Cube Drawing
					std::vector<cv::Point3f> cube;
					for (const auto& p : points) {
						cube.push_back(camera.projectPoint(p));
					}

					{
						// Poor's man Z-Buffering
						zBuffer.prepareTempBuffers();

						const cv::Point3f& p0 = cube.at(4);
						const cv::Point3f& p1 = cube.at(5);
						const cv::Point3f& p2 = cube.at(6);
						const cv::Point3f& p3 = cube.at(7);
						const float pDepth = (std::min(p0.z, p1.z), std::min(p2.z, p3.z)); // Poor's man approximation

						const cv::Point2f p02d = cv::Point2f(p0.x, p0.y);
						const cv::Point2f p12d = cv::Point2f(p1.x, p1.y);
						const cv::Point2f p22d = cv::Point2f(p2.x, p2.y);
						const cv::Point2f p32d = cv::Point2f(p3.x, p3.y);

						const cv::Point facePoints[1][4] = { { p02d , p12d, p22d, p32d } };
						const cv::Point* ppt[1] = { facePoints[0] };
						int npt[] = { 4 };

						cv::fillPoly(zBuffer.getColorTemp(), ppt, npt, 1, cubeColor, cv::LINE_8);
						cv::fillPoly(zBuffer.getDepthTemp(), ppt, npt, 1, cv::Scalar(pDepth), cv::LINE_8);

						zBuffer.drawTempBuffers();
					}

					for (size_t i = 0; i < cubeLinesNumber; i++)
					{
						zBuffer.drawLine(cube.at(cubeLines[i][0]), cube.at(cubeLines[i][1]), cubeColor, cubeThickness);
					}
				}

				{
					// Animation
					animTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startAnimTime).count();
					for (size_t i = 0; i < corners.size(); i++)
					{
						const float maxHeight = 3.0f;
						const float u = corners[i].x / grid_width;
						const float v = corners[i].y / grid_height;
						const float k = (float)std::sin(CV_PI * (u + v + animTime / 2000.0f)) / 2.0f + 0.5f;
						const float k2 = (float)std::cos(CV_PI * (u + v + animTime / 20000.0f)) / 2.0f + 0.5f;

						const cv::Point3f newPos(corners[i].x, corners[i].y, corners[i].z + s.squareSize * maxHeight * (k + 2.0f * k2) / 3.0f);

						const cv::Point3f projectedPos = camera.projectPoint(newPos);
						const cv::Scalar color(255.0f * (k / 2.0f + 0.5), 255.0f * v, 255.0f * u);

						zBuffer.drawCircle(projectedPos, 2, color, 2);

					}
				}

			}
			break;
			default:
				break;
			}
		}

		// Draw Text above everything
		std::string msg = (mode == CalibrationState::Capturing) ? "100/100" : mode == CalibrationState::Calibrated ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
		cv::Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CalibrationState::Capturing)
		{
			if (s.showUndistorsed)
			{
				msg = cv::format("%d/%d Undist", (int)imagePoints.size(), s.nrFrames);
			}
			else
			{
				msg = cv::format("%d/%d Dist", (int)imagePoints.size(), s.nrFrames);
			}
		}

		cv::putText(view, msg, textOrigin, 1, 1, mode == CalibrationState::Calibrated ? GREEN : RED);

		std::string msg2 = cv::format("Frame %llf ms", dt);
		cv::Point secsOrigin(view.cols - 2 * textSize.width - 10, view.rows - 4 * baseLine - 10);
		cv::putText(view, msg2, secsOrigin, 1, 1, mode == CalibrationState::Calibrated ? GREEN : RED);

		if (mode == CalibrationState::Calibrated && found)
		{
			// If the chessboard is found and we are calibrated, draw axis labels
			cv::putText(view, "x", camera.projectPointNoDepth(xAxisText), 1, 3, RED, 3);
			cv::putText(view, "y", camera.projectPointNoDepth(yAxisText), 1, 3, GREEN, 3);
			cv::putText(view, "z", camera.projectPointNoDepth(zAxisText), 1, 3, BLUE, 3);
		}

		if (!s.suppressBlinking && blinkOutput)
		{
			// Maybe turn off this, it is annoying
			cv::bitwise_not(view, view);
		}

		// Show the result undistorted (this doesn't matter if distortion coefficients are null)
		if (mode == CalibrationState::Calibrated && s.showUndistorsed)
		{
			cv::Mat temp = view.clone();
			if (s.useFisheye)
			{
				cv::fisheye::undistortImage(temp, view, camera.CameraIntrisicMatrix(), camera.DistCoeffs());
			}
			else
			{
				cv::undistort(temp, view, camera.CameraIntrisicMatrix(), camera.DistCoeffs());
			}

		}

		// Input
		cv::imshow("Image View", view);
		char key = (char)cv::waitKey(s.inputCapture.isOpened() ? 50 : s.delay); // in ms

		if (key == ESC_KEY)
		{
			break;
		}

		if (key == 'u' && mode == CalibrationState::Calibrated)
		{
			s.showUndistorsed = !s.showUndistorsed;
		}

		if (s.inputCapture.isOpened() && key == 'g')
		{
			// We are a webcam
			mode = CalibrationState::Capturing;
			imagePoints.clear();
			rms = std::numeric_limits<double>::infinity();
			currRestarts = s.restartAttemps;
			currAttempts = s.nrFrames;
		}

		prevFrame = std::chrono::high_resolution_clock::now();
	}

	// show undistorted images for image lists
	if (s.inputType == Settings::InputType::Image_List && s.showUndistorsed)
	{
		cv::Mat view, rview, map1, map2;

		if (s.useFisheye)
		{
			cv::Mat newCamMat;
			cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera.CameraIntrisicMatrix(), camera.DistCoeffs(), imageSize,
				cv::Matx33d::eye(), newCamMat, 1);
			cv::fisheye::initUndistortRectifyMap(camera.CameraIntrisicMatrix(), camera.DistCoeffs(), cv::Matx33d::eye(), newCamMat, imageSize,
				CV_16SC2, map1, map2);
		}
		else
		{
			initUndistortRectifyMap(
				camera.CameraIntrisicMatrix(), camera.DistCoeffs(), cv::Mat(),
				getOptimalNewCameraMatrix(camera.CameraIntrisicMatrix(), camera.DistCoeffs(), imageSize, 1, imageSize, 0), imageSize,
				CV_16SC2, map1, map2);
		}

		for (size_t i = 0; i < s.imageList.size(); i++)
		{
			view = cv::imread(s.imageList[i], cv::IMREAD_COLOR);
			if (view.empty()) {
				continue;
			}

			cv::remap(view, rview, map1, map2, cv::INTER_LINEAR);
			cv::imshow("Image View", rview);
			char c = (char)cv::waitKey();
			if (c == ESC_KEY || c == 'q' || c == 'Q')
			{
				break;
			}

		}
	}

	return 0;
}