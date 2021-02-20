#include "Settings.h"

#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

// Extended from https://docs.opencv.org/4.2.0/d4/d94/tutorial_camera_calibration.html

void Settings::write(cv::FileStorage& fs) const
{

	fs << "{"
		<< "BoardSize_Width" << boardSize.width
		<< "BoardSize_Height" << boardSize.height
		<< "Square_Size" << squareSize;

	if (releaseObject)
	{
		fs << "Grid_Width" << gridWidth;
	}


	fs << "Calibrate_NrOfFrameToUse" << nrFrames
		<< "Acceptable_Threshold" << acceptableThreshold
		<< "Calibrate_FixAspectRatio" << aspectRatio
		<< "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
		<< "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

		<< "Write_DetectedFeaturePoints" << writePoints
		<< "Write_extrinsicParameters" << writeExtrinsics
		<< "Write_gridPoints" << writeGrid
		<< "Write_outputFileName" << outputFileName

		<< "Show_UndistortedImage" << showUndistorsed

		<< "Input_FlipAroundHorizontalAxis" << flipVertical
		<< "Input_Delay" << delay
		<< "Update_Delay" << delayUpdate
		<< "Input" << input
		<< "Use_Kalman" << useKalmanFilter
		<< "Never_Extrinsic_Guess" << neverUseExtrinsicGuess
		<< "Suppress_Blinking" << suppressBlinking
		<< "Restart_Attemps" << restartAttemps
		<< "Stop_Corner_Fix" << dontUserCornerFix
		<< "}";
}

void Settings::read(const cv::FileNode& node)
{
	node["BoardSize_Width"] >> boardSize.width;
	node["BoardSize_Height"] >> boardSize.height;
	node["Square_Size"] >> squareSize;
	const cv::FileNode gridWidthNode = node["Grid_Width"];
	if (!gridWidthNode.empty())
	{
		gridWidthNode >> gridWidth;
		releaseObject = true;
	}
	else
	{
		gridWidth = squareSize * (float)(boardSize.width - 1);
		releaseObject = false;
	}
	node["Calibrate_NrOfFrameToUse"] >> nrFrames;
	node["Acceptable_Threshold"] >> acceptableThreshold;
	node["Calibrate_FixAspectRatio"] >> aspectRatio;
	node["Write_DetectedFeaturePoints"] >> writePoints;
	node["Write_extrinsicParameters"] >> writeExtrinsics;
	node["Write_gridPoints"] >> writeGrid;
	node["Write_outputFileName"] >> outputFileName;
	node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
	node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
	node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
	node["Show_UndistortedImage"] >> showUndistorsed;
	node["Input"] >> input;
	node["Input_Delay"] >> delay;
	node["Update_Delay"] >> delayUpdate;
	node["Fix_K1"] >> fixK1;
	node["Fix_K2"] >> fixK2;
	node["Fix_K3"] >> fixK3;
	node["Fix_K4"] >> fixK4;
	node["Fix_K5"] >> fixK5;
	node["Use_Kalman"] >> useKalmanFilter;
	node["Never_Extrinsic_Guess"] >> neverUseExtrinsicGuess;
	const cv::FileNode suppressBlinkingNode = node["Suppress_Blinking"];
	if (!suppressBlinkingNode.empty())
	{
		suppressBlinkingNode >> suppressBlinking;
	}
	else
	{
		suppressBlinking = false;
	}
	node["Restart_Attemps"] >> restartAttemps;
	const cv::FileNode dontUserCornerFixNode = node["Stop_Corner_Fix"];
	if (!dontUserCornerFixNode.empty())
	{
		dontUserCornerFixNode >> dontUserCornerFix;
	}
	else
	{
		dontUserCornerFix = false;
	}
	validate();
}

void Settings::validate()
{
	goodInput = true;
	if (boardSize.width <= 0 || boardSize.height <= 0)
	{
		std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
		goodInput = false;
	}
	if (squareSize <= 10e-6)
	{
		std::cerr << "Invalid square size " << squareSize << std::endl;
		goodInput = false;
	}
	if (gridWidth < squareSize)
	{
		std::cerr << "Invalid grid width " << gridWidth << std::endl;
		goodInput = false;
	}
	if (nrFrames <= 0)
	{
		std::cerr << "Invalid number of frames " << nrFrames << std::endl;
		goodInput = false;
	}

	if (input.empty())      // Check for valid input
	{
		inputType = InputType::Invalid;
	}
	else
	{
		if (input[0] >= '0' && input[0] <= '9')
		{
			std::stringstream ss(input);
			ss >> cameraID;
			inputType = InputType::Camera;
		}
		else
		{
			if (isListOfImages(input) && readStringList(input, imageList))
			{
				inputType = InputType::Image_List;
				nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
			}
			else
			{
				inputType = InputType::Video_File;
			}
		}
		switch (inputType)
		{
		case InputType::Camera:
			inputCapture.open(cameraID);
			break;
		case InputType::Video_File:
			inputCapture.open(input);
			break;
		default:
			break;
		}
		if (inputType != InputType::Image_List && !inputCapture.isOpened())
		{
			inputType = InputType::Invalid;
		}
	}
	if (inputType == InputType::Invalid)
	{
		std::cerr << " Input does not exist: " << input;
		goodInput = false;
	}

	flag = 0;
	if (calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
	if (calibZeroTangentDist)   flag |= cv::CALIB_ZERO_TANGENT_DIST;
	if (aspectRatio)            flag |= cv::CALIB_FIX_ASPECT_RATIO;
	if (fixK1)                  flag |= cv::CALIB_FIX_K1;
	if (fixK2)                  flag |= cv::CALIB_FIX_K2;
	if (fixK3)                  flag |= cv::CALIB_FIX_K3;
	if (fixK4)                  flag |= cv::CALIB_FIX_K4;
	if (fixK5)                  flag |= cv::CALIB_FIX_K5;

	atImageList = 0;

}

cv::Mat Settings::nextImage()
{
	cv::Mat result;
	if (inputCapture.isOpened())
	{
		cv::Mat view0;
		inputCapture >> view0;
		view0.copyTo(result);
	}
	else if (atImageList < imageList.size())
	{
		result = cv::imread(imageList[atImageList++], cv::IMREAD_COLOR);
	}

	return result;
}

bool Settings::readStringList(const std::string& filename, std::vector<std::string>& l)
{
	l.clear();

	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		return false;
	}

	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
	{
		return false;
	}

	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		l.push_back((std::string)*it);
	}

	return true;
}

bool Settings::isListOfImages(const std::string& filename)
{
	std::string s(filename);
	// Look for file extension
	if (s.find(".xml") == std::string::npos && s.find(".yaml") == std::string::npos && s.find(".yml") == std::string::npos)
	{
		return false;
	}
	else
	{
		return true;
	}
}