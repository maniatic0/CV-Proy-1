#pragma once

#include <limits>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Poor's man Z-Buffer
class ZBuffer {

	cv::Mat colorBuff;
	cv::Mat depthBuff;
	cv::Mat colorTemp;
	cv::Mat depthTemp;

public:
	/// <summary>
	/// Value to clean Depth Buffers
	/// </summary>
	constexpr static float depthClean = std::numeric_limits<float>::infinity();

	/// <summary>
	/// Reset the Depth Buffer
	/// </summary>
	inline void resetDepthBuffer()
	{
		depthBuff.create(colorBuff.rows, colorBuff.cols, CV_32FC1);
		depthBuff.setTo(cv::Scalar(depthClean));
	}

	/// <summary>
	/// Prepare Temp Buffers
	/// </summary>
	inline void prepareTempBuffers()
	{
		resetTempBuffers(colorBuff, depthBuff, colorTemp, depthTemp);;
	}

	/// <summary>
	/// Perform Z Buffering
	/// </summary>
	inline void drawTempBuffers()
	{
		zBuffering(colorBuff, depthBuff, colorTemp, depthTemp);;
	}

	/// <summary>
	/// Set Color Buffer
	/// </summary>
	/// <typeparam name="T">Must be cv::Mat</typeparam>
	/// <param name="color_">Buffer to use</param>
	template<typename T = cv::Mat>
	inline void setColor(T&& color_)
	{
		colorBuff = color_;
		resetDepthBuffer();
	}

	inline cv::Mat& getColor()
	{
		return colorBuff;
	}

	inline cv::Mat& getColorTemp()
	{
		return colorTemp;
	}

	inline cv::Mat& getDepthTemp()
	{
		return depthTemp;
	}

	// See cv::line. Note that pt1 and pt2 must contain their depthBuff in the last position
	inline void drawLine(cv::Point3f pt1, cv::Point3f pt2, const cv::Scalar& color,
		int thickness = 1, int lineType = cv::LINE_8, int shift = 0)
	{
		// Poor's man Z-Buffering
		prepareTempBuffers();

		cv::Point2f ori(pt1.x, pt1.y);
		cv::Point2f dst(pt2.x, pt2.y);
		const float pDepth = std::min(pt1.z, pt2.z); // Poor's man approximation

		cv::line(colorTemp, ori, dst, color, thickness, lineType, shift);
		cv::line(depthTemp, ori, dst, pDepth, thickness, lineType, shift);

		drawTempBuffers();
	}

	inline void drawCircle(cv::Point3f center, int radius,
		const cv::Scalar& color, int thickness = 1,
		int lineType = cv::LINE_8, int shift = 0)
	{
		// Poor's man Z-Buffering
		prepareTempBuffers();

		const cv::Point2f center2d(center.x, center.y);

		cv::circle(colorTemp, center2d, radius, color, thickness, lineType, shift);
		cv::circle(depthTemp, center2d, radius, cv::Scalar(center.z), thickness, lineType, shift);

		drawTempBuffers();
	}

	/// <summary>
	/// Clean 
	/// Temp Depth Buffers
	/// </summary>
	/// <param name="colorBuff">Color Buffer</param>
	/// <param name="depthBuff">Depth Buffer</param>
	/// <param name="colorTemp">Temp Color Buffer</param>
	/// <param name="depthTemp">Temp Depth Buffer</param>
	inline static void resetTempBuffers(const cv::Mat& view, const cv::Mat& depth, cv::Mat& colorTemp, cv::Mat& depthTemp)
	{
		colorTemp.create(view.rows, view.cols, view.type());
		colorTemp.setTo(cv::Scalar(0, 0, 0));

		depthTemp.create(view.rows, view.cols, depth.type());
		depthTemp.setTo(cv::Scalar(depthClean));
	}

	/// <summary>
	/// Peform Z Buffering
	/// </summary>
	/// <param name="colorBuff">Color Buffer</param>
	/// <param name="depthBuff">Depth Buffer</param>
	/// <param name="colorTemp">Temp Color Buffer</param>
	/// <param name="depthTemp">Temp Depth Buffer</param>
	inline static void zBuffering(cv::Mat& view, cv::Mat& depth, const cv::Mat& colorTemp, const cv::Mat& depthTemp)
	{
		for (int i = 0; i < view.rows; i++)
		{
			for (int j = 0; j < view.cols; j++)
			{
				if (depthTemp.at<float>(i, j) < depth.at<float>(i, j))
				{
					view.at<cv::Vec3b>(i, j) = colorTemp.at<cv::Vec3b>(i, j);
					depth.at<float>(i, j) = depthTemp.at<float>(i, j);
				}
			}
		}
	}
};