#include "CameraCalibration.h"
#include "Settings.h"

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

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

	float grid_width = s.squareSize * (s.boardSize.width - 1);
	bool release_object = false;
	if (parser.has("d")) {
		grid_width = parser.get<float>("d");
		release_object = true;
	}

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	CalibrationState mode = s.inputType == Settings::InputType::IMAGE_LIST ? CalibrationState::CAPTURING : CalibrationState::DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;

	//! [get_input]
	while (true)
	{
		Mat view;
		bool blinkOutput = false;

		view = s.nextImage();

		//-----  If no more image, or got enough, then stop calibration and show result -------------
		if (mode == CalibrationState::CAPTURING && imagePoints.size() >= (size_t)s.nrFrames)
		{
			if (runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints, grid_width,
				release_object))
			{
				mode = CalibrationState::CALIBRATED;
			}
			else
			{
				mode = CalibrationState::DETECTION;
			}
		}
		if (view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if (mode != CalibrationState::CALIBRATED && !imagePoints.empty())
			{
				runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints, grid_width,
					release_object);
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

			if (mode == CalibrationState::CAPTURING &&  // For camera only take new samples after delay time
				(!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay * 1e-3 * CLOCKS_PER_SEC))
			{
				imagePoints.push_back(pointBuf);
				prevTimestamp = clock();
				blinkOutput = s.inputCapture.isOpened();
			}

			// Draw the corners.
			drawChessboardCorners(view, s.boardSize, Mat(pointBuf), found);
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
				msg = format("%d/%d", (int)imagePoints.size(), s.nrFrames);
			}
		}

		putText(view, msg, textOrigin, 1, 1, mode == CalibrationState::CALIBRATED ? GREEN : RED);

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
				cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
			}
			else
			{
				undistort(temp, view, cameraMatrix, distCoeffs);
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
		}
		//! [await_input]
	}

	// -----------------------Show the undistorted image for the image list ------------------------
	//! [show_results]
	if (s.inputType == Settings::InputType::IMAGE_LIST && s.showUndistorsed)
	{
		Mat view, rview, map1, map2;

		if (s.useFisheye)
		{
			Mat newCamMat;
			fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
				Matx33d::eye(), newCamMat, 1);
			fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,
				CV_16SC2, map1, map2);
		}
		else
		{
			initUndistortRectifyMap(
				cameraMatrix, distCoeffs, Mat(),
				getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
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