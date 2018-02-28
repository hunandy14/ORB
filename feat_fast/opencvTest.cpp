#include <iostream>
#include <vector>
#include "opencvTest.hpp"
using namespace std;


void drawCornerOnImage(Mat & image, const Mat &binary){
	Mat_ < uchar > ::const_iterator it = binary.begin < uchar >();
	Mat_ < uchar > ::const_iterator itd = binary.end < uchar >();
	for(int i = 0; it != itd; it++, i++){
		if(*it)
			circle(image, Point(i % image.cols, i / image.cols), 3, Scalar(0, 255, 0), 1);
	}
}
void opencvHarris(){
	Mat image = imread("kanna.bmp");
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	Mat cornerStrength;
	cornerHarris(gray, cornerStrength, 3, 3, 0.01);

	double maxStrength;
	double minStrength;
	// 找到影象中的最大、最小值
	minMaxLoc(cornerStrength, &minStrength, &maxStrength);

	Mat dilated;
	Mat locaMax;
	// 膨脹影象，最找出影象中全部的區域性最大值點
	dilate(cornerStrength, dilated, Mat());
	// compare是一個邏輯比較函式，返回兩幅影象中對應點相同的二值影象
	compare(cornerStrength, dilated, locaMax, CMP_EQ);

	Mat cornerMap;
	double qualityLevel = 0.01;
	double th = qualityLevel * maxStrength; // 閾值計算
	threshold(cornerStrength, cornerMap, th, 255, THRESH_BINARY);
	cornerMap.convertTo(cornerMap, CV_8U);
	// 逐點的位運算
	bitwise_and(cornerMap, locaMax, cornerMap);


	drawCornerOnImage(image, cornerMap);

	namedWindow("result");
	imshow("result", image);
	waitKey();
}
void opencvHarris2(){
	cv::Mat image_color = cv::imread("kanna.bmp", cv::IMREAD_COLOR);  

	//使用灰度图像进行角点检测  
	cv::Mat image_gray;  
	cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);  

	//设置角点检测参数  
	std::vector<cv::Point2f> corners;  
	int max_corners = 200;  
	double quality_level = 0.01;  
	double min_distance = 3.0;  
	int block_size = 3;  
	bool use_harris = false;  
	double k = 0.04;  

	//角点检测  
	cv::goodFeaturesToTrack(image_gray,   
		corners,   
		max_corners,   
		quality_level,   
		min_distance,   
		cv::Mat(),   
		block_size,   
		use_harris,   
		k);  

	//将检测到的角点绘制到原图上  
	for (int i = 0; i < corners.size(); i++)  
	{  
		cv::circle(image_color, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);  
	}  

	cv::imshow("house corner", image_color);  
	cv::waitKey(0);  
}
void OpencvFast(string filename){
	Mat img_1 = imread(filename);
	if(!img_1.data)
		throw runtime_error("No File.");
	cv::Ptr<Feature2D> detector = FastFeatureDetector::create();
	std::vector<KeyPoint> keypoints_1;
	detector->detect(img_1, keypoints_1);
	drawKeypoints(img_1, keypoints_1, img_1, Scalar::all(255));
	imshow("fast", img_1);

	std::vector<char> mask;

	waitKey(0);
}