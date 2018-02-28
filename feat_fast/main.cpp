#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#include "Raw2Img\Raw2img.hpp"
#include "Imgraw\Imgraw.hpp"

extern "C" {
#include "fastlib\fast.h"
}

#include <opencv2\opencv.hpp> 
using namespace cv;

#define Fast_THRE 10



void harris(ImgRaw & img, xy* & feat_harris, int* numcorners_harris, const xy * feat, int numcorners);

void OpencvFast(string filename);
void FastTest(){
	// 讀寫 Bmp
	vector<unsigned char> raw_img;
	uint32_t weidth, heigh;
	uint16_t bits;
	Raw2Img::read_bmp(raw_img, "kanna.bmp", &weidth, &heigh, &bits);
	Raw2Img::raw2gray(raw_img);
	// FAST 找特徵點
	const unsigned char* data = raw_img.data();
	int xsize = weidth, ysize = heigh, stride = weidth, threshold = 10, numcorners;
	xy* feat_point;
	feat_point = fast9_detect_nonmax(data, xsize, ysize, stride, threshold, &numcorners);
	// 輸出到圖片
	cout << "numcorners=" << numcorners << endl;
	for(size_t i = 0; i < numcorners; i++){
		int& x = feat_point[i].x;
		int& y = feat_point[i].y;
		raw_img[y*weidth + x] = 0;
	}
	Raw2Img::raw2bmp("out.bmp", raw_img, weidth, heigh, 8);
}

vector<double> GWCM(const byte* im, int xsize, int ysize, int stride, 
	xy* feat_point, int ret_num_corners, int radius);

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
//====================================================================================
int main(int argc, char const *argv[]){
	// 讀寫 Bmp
	ImgRaw img1("kanna.bmp");
	img1 = img1.ConverGray();
	// FAST 找特徵點
	vector<unsigned char> raw_data = img1;
	const unsigned char* data = raw_data.data();
	int xsize = img1.width, ysize = img1.height, stride = xsize, threshold = 10, numcorners;
	xy* feat;
	feat = fast9_detect_nonmax(data, xsize, ysize, stride, threshold, &numcorners);
	// 輸出到圖片
	cout << "numcorners=" << numcorners << endl;
	ImgRaw temp = img1;
	for(size_t i = 0; i < numcorners; i++){
		int& x = feat[i].x;
		int& y = feat[i].y;
		temp.at2d(y, x) = 0;
	}
	temp.bmp("FAST.bmp");


	// Harris 角點偵測
	xy* feat_harris = nullptr;
	int numcorners_harris = 0;
	harris(img1, feat_harris, &numcorners_harris, feat, numcorners);



	// 輸出到圖片
	cout << "numcorners_harris=" << numcorners_harris << endl;
	temp = img1;
	for(size_t i = 0; i < numcorners_harris; i++){
		int& x = feat_harris[i].x;
		int& y = feat_harris[i].y;
		//temp.at2d(y, x) = 0;
	}
	temp.bmp("harris.bmp");

	//opencvHarris();

	auto sing = GWCM(data, xsize, ysize, stride, 
		feat, numcorners, 3);

	temp = img1;
	for(size_t i = 0; i < numcorners; i++){
		Draw::draw_arrow(temp, feat[i].y, feat[i].x, 100, sing[i]);
	}
	temp.bmp("arrow.bmp");

	
	return 0;
}

#define HarrisR 2
void harris(ImgRaw& img, xy* & feat_harris, int* numcorners_harris, const xy* feat, int numcorners){
	const int r = HarrisR;
	feat_harris = new xy[img.width*img.height]{};
	int& idx = *numcorners_harris = 0;

	vector<bool> h(numcorners);
	vector<float> cornerStrength(numcorners);

	for(int k = 0; k < numcorners; k++){
		const int& i = feat[k].x;
		const int& j = feat[k].y;

		int A = 0, B = 0, C = 0;
		for(int rj = -r+1; rj < r-1; rj++){
			for(int ri = -r+1; ri < r-1; ri++){
				float dx=img.at2d(j+rj, i+ri+1) - img.at2d(j+rj, i+ri-1);
				float dy=img.at2d(j+rj+1, i+ri) - img.at2d(j+rj-1, i+ri);
				A += pow(dx, 2);
				B += pow(dy, 2);
				C += dy*dx;
			}
		}
		int detM = A*C + B*B;
		int traceM = A+C;

		const float afa=0.04, t=0.01;
		const float data = detM - afa*traceM*traceM;
		if(data > t){
			//feat_harris[idx] = xy{j, i};
			h[idx] = 1;
			cornerStrength[idx]=data;
			++idx;
		}
	}

	// 過濾周圍
	for(int j = 0, c=0; j < img.height; j++){
		for(int i = 0; i < img.width; i++, c++){
			
			int idxp = (j-1)*img.width+i;
			int idx = j*img.width+i;
			int idxn = (j+1)*img.width+i;
			if(h[idx] == 1){
				if(cornerStrength[idx] > cornerStrength[idx+1] and
					cornerStrength[idx] > cornerStrength[idx-1] and

					cornerStrength[idx] > cornerStrength[idxp-1] and
					cornerStrength[idx] > cornerStrength[idxp+0] and
					cornerStrength[idx] > cornerStrength[idxp+1] and

					cornerStrength[idx] > cornerStrength[idxn-1] and
					cornerStrength[idx] > cornerStrength[idxn+0] and
					cornerStrength[idx] > cornerStrength[idxn+1]
					){
					h[idx] = 1;
				} else{
					h[idx] = 0;
				}
					feat_harris[idx] = xy{j, i};
			}
		}
		if(c >= idx){
			break;
		}
	}


	//OpencvFast("kanna.bmp");
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
//====================================================================================
//灰度重心法
vector<double> GWCM(const byte* im, int xsize, int ysize, int stride, 
	xy* feat_point, int ret_num_corners, int radius)
{
	int point_nub = 0;
	int m10, m01;
	vector<double> feat_sita(ret_num_corners);

	// m10
	m10 = 0;
	while (point_nub <= ret_num_corners)
	{
		// m10
		m10 = 0;
		for (int i = -radius; i < radius; i++)
		{
			int thisx = feat_point[point_nub].x + i;

			if(thisx < 0.0 || thisx >= xsize){

				//cout << "特徵點太過靠近邊緣" << endl;
			}
			else
			{
				const byte* p = im + feat_point[point_nub].y * stride + thisx;
				m10 += i * (int)p;
			}
		}

		// m01
		m01 = 0;
		for (int j = -radius; j < radius; j++)
		{
			int thisy = thisy = feat_point[point_nub].y + j;

			if(thisy + j < 0.0 || thisy + j < ysize){

				//cout << "特徵點太過靠近邊緣" << endl;
			}
			else
			{
				const byte* p = im + thisy * stride + feat_point[point_nub].x;
				m01 += j * (int)p;
			}
		}
		feat_sita[point_nub++] = fastAtan2(m01, m10);
	}
	return feat_sita;
}