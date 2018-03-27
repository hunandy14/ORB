/*****************************************************************
Name : 
Date : 2018/03/01
By   : CharlotteHonG
Final: 2018/03/19
*****************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>
using namespace std;
#include <opencv2\opencv.hpp> 
using namespace cv;
#include <Timer.hpp>
//====================================================================================
#include "Imgraw.hpp"
#include "feat.hpp"
#include "opencvTest.hpp"

#include "stitch\imagedata.hpp"
#include "stitch\Blend.hpp"

#include "ORB.hpp"

//====================================================================================
int main(int argc, char const *argv[]) {
	//#define harrisTest
#ifdef harrisTest
	opencvHarris3();
#else
	string name1 = "sc02.bmp", name2= "sc03.bmp";
	//string name1 = "b01.bmp", name2= "b02.bmp";
	// 開圖
	ImgRaw img1(name1, "", 0);

	ImgRaw img1_gray = img1.ConverGray();
	img1_gray.nomal=0;

	Timer t;
	// ORB
	Feat feat;
	create_ORB(img1_gray, feat);

	// 開圖
	ImgRaw img2(name2, "", 0);
	ImgRaw img2_gray = img2.ConverGray();
	img2_gray.nomal=0;

	// ORB
	Feat feat2;
	create_ORB(img2_gray, feat2);

	// 尋找配對點
	vector<float> HomogMat;
	matchORB(feat2, feat, HomogMat);

	// 測試配對點
	ImgRaw stackImg = imgMerge(img1, img2);
	//stackImg.bmp("merge.bmp");
	//featDrawLine("line.bmp", stackImg, feat2);

	// 縫合圖片
	ImgRaw imgL(name1);
	ImgRaw imgR(name2);

	size_t RANSAC_num=0;
	Feature** RANSAC_feat=nullptr;
	//RANSAC_feat = new Feature*[RANSAC_num];
	getNewfeat(feat2, RANSAC_feat, RANSAC_num);
	//featDrawLine2("_matchImg_RANSACImg.bmp", stackImg, RANSAC_feat, RANSAC_num);
	blen2img(imgL, imgR, HomogMat, RANSAC_feat, RANSAC_num);
	t.print("All time");


#endif // harrisTest
	return 0;
}
//====================================================================================