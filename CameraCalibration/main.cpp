#include "CameraCalibration.h"
#include "Settings.h"
#include "Camera.h"

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	const String keys
		= "{help h usage ? |           | print this message            }"
		"{@settings      |default.xml| input setting file            }"
		"{d              |           | actual distance between top-left and top-right corners of "
		"the calibration grid }"
		"{winSize        | 11        | Half of search window for cornerSubPix }";
	CommandLineParser parser(argc, argv, keys);
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

	//! [file_read]
	Settings s;
	{
		const string inputSettingsFile = parser.get<string>(0);
		FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
		if (!fs.isOpened())
		{
			cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
			parser.printMessage();
			return -1;
		}
		if (!Settings::readFromFile(fs, s))
		{
			fs.release(); // close Settings file
			cout << "Invalid input detected. Application stopping. " << endl;
			return -1;
		}
		fs.release(); // close Settings file
	}
	//! [file_read]

	//FileStorage fout("settings.yml", FileStorage::WRITE); // write config as YAML
	//fout << "Settings" << s;

	int winSize = parser.get<int>("winSize");

	float grid_width = s.gridWidth;
	bool release_object = s.releaseObject;
	if (parser.has("d")) {
		grid_width = parser.get<float>("d");
		release_object = true;
	}

	std::vector<cv::Point3f> corners;
	calcBoardCornerPositions(s.boardSize, s.squareSize, corners, s.calibrationPattern);

	float grid_height = s.squareSize * (float)(s.boardSize.height - 1);

	Camera camera;
	cv::Mat inliers;
	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrixTemp, distCoeffsTemp;
	Mat rvecTemp, tvecTemp;
	double rms = std::numeric_limits<double>::infinity();
	double rmsTemp;
	bool atLeastOneSuccesss = false;
	size_t preImagePointsSize = imagePoints.size();
	Size imageSize;
	CalibrationState mode = s.inputType == Settings::InputType::IMAGE_LIST ? CalibrationState::CAPTURING : CalibrationState::DETECTION;
	clock_t prevTimestamp = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> prevFrame = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> startAnimTime = std::chrono::high_resolution_clock::now();
	double dt;
	double animTime = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0), BLUE(255, 0, 0); // BGR

	const char ESC_KEY = 27;

	// For Axis Purposes
	Point3f origin(0, 0, 0);
	int axisThickness = 3;
	float axisMultiplier = 4.0f;
	Point3f xAxis(s.squareSize * axisMultiplier, 0, 0);
	Point3f yAxis(0, s.squareSize * axisMultiplier, 0);
	Point3f zAxis(0, 0, s.squareSize * axisMultiplier);
	float axisTextOffset = 2.0f;
	Point3f xAxisText(s.squareSize * axisMultiplier + axisTextOffset, 0, 0);
	Point3f yAxisText(0, s.squareSize * axisMultiplier + axisTextOffset, 0);
	Point3f zAxisText(0, 0, s.squareSize * axisMultiplier + axisTextOffset);

	// For Cube
	vector<Point3d> points;
	points.push_back(Point3d(0, 0, 0));
	points.push_back(Point3d(0, 50, 0));
	points.push_back(Point3d(0, 50, 50));
	points.push_back(Point3d(0, 0, 50));
	points.push_back(Point3d(50, 0, 0));
	points.push_back(Point3d(50, 50, 0));
	points.push_back(Point3d(50, 50, 50));
	points.push_back(Point3d(50, 0, 50));
	cv:Scalar cubeColor = Scalar(255, 0, 0);
	int cubeThickness = 7;

	//! [get_input]
	while (true)
	{
		dt = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - prevFrame).count();

		Mat view;
		bool blinkOutput = false;

		view = s.nextImage();

		//-----  If no more image, or got enough, then try to calibrate and show result -------------
		if (mode == CalibrationState::CAPTURING && imagePoints.size() >= 1 && preImagePointsSize < imagePoints.size())
		{
			const CalibrationResult res = runCalibrationAndSave(s, imageSize, cameraMatrixTemp, distCoeffsTemp, rvecTemp, tvecTemp, imagePoints, corners, grid_width,
				release_object, rmsTemp, rms, !atLeastOneSuccesss);
			if (res != CalibrationResult::FAILED)
			{
				atLeastOneSuccesss = true;
				if (res == CalibrationResult::SUCCESS)
				{
					// We are better
					rms = rmsTemp;
					camera.setIntrinsics(cameraMatrixTemp, distCoeffsTemp);
					camera.setExtrinsics(rvecTemp, tvecTemp);


					if (imagePoints.size() >= (size_t)s.nrFrames)
					{
						// We are done by Number of Frames
						mode = CalibrationState::CALIBRATED;
						std::cout << "We are calibrated by Number of Frames!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
					}
					else if (rms <= s.acceptableThreshold)
					{
						// We are done by Threshold
						mode = CalibrationState::CALIBRATED;
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
				// mode = CalibrationState::DETECTION; // ?
				std::cout << "We failed to calibrate!" << std::endl;
			}
		}
		preImagePointsSize = imagePoints.size();
		if (view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if (!imagePoints.empty())
			{
				const CalibrationResult res = runCalibrationAndSave(s, imageSize, cameraMatrixTemp, distCoeffsTemp, rvecTemp, tvecTemp, imagePoints, corners, grid_width,
					release_object, rmsTemp, rms, !atLeastOneSuccesss);
				if (res != CalibrationResult::FAILED)
				{
					atLeastOneSuccesss = true;
					if (res == CalibrationResult::SUCCESS)
					{
						// We are better
						rms = rmsTemp;
						camera.setIntrinsics(cameraMatrixTemp, distCoeffsTemp);
						camera.setExtrinsics(rvecTemp, tvecTemp);

						if (imagePoints.size() >= (size_t)s.nrFrames)
						{
							// We are done
							mode = CalibrationState::CALIBRATED;
							std::cout << "We are calibrated!" << std::endl;
						}
						else
						{
							std::cout << "We improve the calibration!: " << imagePoints.size() << "/" << (size_t)s.nrFrames << std::endl;
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
		//! [get_input]

		imageSize = view.size();  // Format input image.
		if (s.flipVertical)
		{
			flip(view, view, 0);
		}

		//! [find_pattern]
		vector<Point2f> pointBuf;

		bool found;

		int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

		if (!s.useFisheye) {
			// fast check erroneously fails with high distortions like fisheye
			chessBoardFlags |= CALIB_CB_FAST_CHECK;
		}

		switch (s.calibrationPattern) // Find feature points on the input format
		{
		case Settings::Pattern::CHESSBOARD:
			found = findChessboardCorners(view, s.boardSize, pointBuf, chessBoardFlags);
			break;
		case Settings::Pattern::CIRCLES_GRID:
			found = findCirclesGrid(view, s.boardSize, pointBuf);
			break;
		case Settings::Pattern::ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid(view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			found = false;
			break;
		}
		//! [find_pattern]
		//! [pattern_found]
		if (found)                // If done with success,
		{
			// improve the found corners' coordinate accuracy for chessboard
			if (s.calibrationPattern == Settings::Pattern::CHESSBOARD)
			{
				Mat viewGray;
				cvtColor(view, viewGray, COLOR_BGR2GRAY);
				cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
					Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
			}

			// Draw the corners.
			drawChessboardCorners(view, s.boardSize, Mat(pointBuf), found);

			switch (mode)
			{
			case CalibrationState::CAPTURING:
			{
				if (!s.inputCapture.isOpened() || clock() - prevTimestamp > (clock_t)s.delay * 1e-3 * CLOCKS_PER_SEC)
				{
					// For camera only take new samples after delay time
					imagePoints.push_back(pointBuf);
					prevTimestamp= clock();
					blinkOutput = s.inputCapture.isOpened();
				}
			}
			break;
			case CalibrationState::CALIBRATED:
			{
				if (clock() - prevTimestamp > (clock_t)s.delayUpdate * 1e-3 * CLOCKS_PER_SEC)
				{
					// For camera only take new samples after delay time
					camera.estimatePose(corners, pointBuf, (double)(clock() - prevTimestamp) / (double)CLOCKS_PER_SEC, inliers);
					prevTimestamp = clock();
				}
				
				{
					// Axis Drawing
					cv::Point2f ori = camera.projectPoint(origin);
					cv::Point2f xAxis2d = camera.projectPoint(xAxis);
					cv::Point2f yAxis2d = camera.projectPoint(yAxis);
					cv::Point2f zAxis2d = camera.projectPoint(zAxis);

					cv::line(view, ori, xAxis2d, RED, axisThickness);
					cv::line(view, ori, yAxis2d, GREEN, axisThickness);
					cv::line(view, ori, zAxis2d, BLUE, axisThickness);

					// Removed due to inverted axis
					// cv::drawFrameAxes(view, camera.CameraMatrix(), camera.DistCoeffs(), camera.RotationVec(), camera.TranslationVec(), s.squareSize * 2.0f);
				}
				

				{
					// Cube Drawing
					vector<Point2f> cube;
					for (const auto &p : points) {
						cube.push_back(camera.projectPoint(p));
					}

					cv::line(view, cube.at(0), cube.at(1), cubeColor, cubeThickness);
					cv::line(view, cube.at(1), cube.at(2), cubeColor, cubeThickness);
					cv::line(view, cube.at(2), cube.at(3), cubeColor, cubeThickness);
					cv::line(view, cube.at(3), cube.at(0), cubeColor, cubeThickness);
					cv::line(view, cube.at(0), cube.at(4), cubeColor, cubeThickness);
					cv::line(view, cube.at(1), cube.at(5), cubeColor, cubeThickness);
					cv::line(view, cube.at(2), cube.at(6), cubeColor, cubeThickness);
					cv::line(view, cube.at(3), cube.at(7), cubeColor, cubeThickness);
					cv::line(view, cube.at(4), cube.at(5), cubeColor, cubeThickness);
					cv::line(view, cube.at(5), cube.at(6), cubeColor, cubeThickness);
					cv::line(view, cube.at(6), cube.at(7), cubeColor, cubeThickness);
					cv::line(view, cube.at(7), cube.at(4), cubeColor, cubeThickness);
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
						cv::Point3f newPos(corners[i].x, corners[i].y, corners[i].z + s.squareSize * maxHeight * (k + 2.0f * k2) / 3.0f);
						cv::Scalar color(255.0f * (k / 2.0f + 0.5), 255.0f * v, 255.0f * u);
						cv::circle(view, camera.projectPoint(newPos), 2, color, 2);
					}
				}
				
			}
			break;
			default:
				break;
			}
		}
		//! [pattern_found]
		//----------------------------- Output Text ------------------------------------------------
		//! [output_text]
		string msg = (mode == CalibrationState::CAPTURING) ? "100/100" :
			mode == CalibrationState::CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CalibrationState::CAPTURING)
		{
			if (s.showUndistorsed)
			{
				msg = format("%d/%d Undist", (int)imagePoints.size(), s.nrFrames);
			}
			else
			{
				msg = format("%d/%d Dist", (int)imagePoints.size(), s.nrFrames);
			}
		}

		putText(view, msg, textOrigin, 1, 1, mode == CalibrationState::CALIBRATED ? GREEN : RED);

		string msg2 = format("Frame %llf ms", dt);
		Point secsOrigin(view.cols - 2 * textSize.width - 10, view.rows - 4 * baseLine - 10);
		putText(view, msg2, secsOrigin, 1, 1, mode == CalibrationState::CALIBRATED ? GREEN : RED);

		if (mode == CalibrationState::CALIBRATED)
		{
			if (found)
			{
				putText(view, "x", camera.projectPoint(xAxisText), 1, 3, RED, 3);
				putText(view, "y", camera.projectPoint(yAxisText), 1, 3, GREEN, 3);
				putText(view, "z", camera.projectPoint(zAxisText), 1, 3, BLUE, 3);
			}

		}

		if (blinkOutput)
		{
			bitwise_not(view, view);
		}

		//! [output_text]
		//------------------------- Video capture  output  undistorted ------------------------------
		//! [output_undistorted]
		if (mode == CalibrationState::CALIBRATED && s.showUndistorsed)
		{
			Mat temp = view.clone();
			if (s.useFisheye)
			{
				cv::fisheye::undistortImage(temp, view, camera.CameraMatrix(), camera.DistCoeffs());
			}
			else
			{
				undistort(temp, view, camera.CameraMatrix(), camera.DistCoeffs());
			}

		}
		//! [output_undistorted]
		//------------------------------ Show image and check for input commands -------------------
		//! [await_input]
		imshow("Image View", view);
		char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

		if (key == ESC_KEY)
		{
			break;
		}

		if (key == 'u' && mode == CalibrationState::CALIBRATED)
		{
			s.showUndistorsed = !s.showUndistorsed;
		}

		if (s.inputCapture.isOpened() && key == 'g')
		{
			mode = CalibrationState::CAPTURING;
			imagePoints.clear();
			rms = std::numeric_limits<double>::infinity();
		}
		//! [await_input]
		prevFrame = std::chrono::high_resolution_clock::now();
	}

	// -----------------------Show the undistorted image for the image list ------------------------
	//! [show_results]
	if (s.inputType == Settings::InputType::IMAGE_LIST && s.showUndistorsed)
	{
		Mat view, rview, map1, map2;

		if (s.useFisheye)
		{
			Mat newCamMat;
			fisheye::estimateNewCameraMatrixForUndistortRectify(camera.CameraMatrix(), camera.DistCoeffs(), imageSize,
				Matx33d::eye(), newCamMat, 1);
			fisheye::initUndistortRectifyMap(camera.CameraMatrix(), camera.DistCoeffs(), Matx33d::eye(), newCamMat, imageSize,
				CV_16SC2, map1, map2);
		}
		else
		{
			initUndistortRectifyMap(
				camera.CameraMatrix(), camera.DistCoeffs(), Mat(),
				getOptimalNewCameraMatrix(camera.CameraMatrix(), camera.DistCoeffs(), imageSize, 1, imageSize, 0), imageSize,
				CV_16SC2, map1, map2);
		}

		for (size_t i = 0; i < s.imageList.size(); i++)
		{
			view = imread(s.imageList[i], IMREAD_COLOR);
			if (view.empty()) {
				continue;
			}

			remap(view, rview, map1, map2, INTER_LINEAR);
			imshow("Image View", rview);
			char c = (char)waitKey();
			if (c == ESC_KEY || c == 'q' || c == 'Q')
			{
				break;
			}

		}
	}
	//! [show_results]

	return 0;
}