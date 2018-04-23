/*****************************************************************
Name : 
Date : 2018/03/01
By   : CharlotteHonG
Final: 2018/03/19
*****************************************************************/
#pragma once

#include "Imgraw.hpp"
#include "feat\feat.hpp"
void harris_coners(const ImgRaw& img, Feat& feat);


#include <opencv2\opencv.hpp> 
void goodFeaturesToTrack2( cv::InputArray image, cv::OutputArray corners,
	int maxCorners, double qualityLevel, double minDistance,
	cv::InputArray mask, int blockSize,
	int gradientSize, bool useHarrisDetector = false,
	double k = 0.04 );