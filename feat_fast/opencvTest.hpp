#pragma once

#include <opencv2\opencv.hpp> 
using namespace cv;

void drawCornerOnImage(Mat & image, const Mat & binary);
void opencvHarris();
void opencvHarris2();
void OpencvFast(std::string filename);
