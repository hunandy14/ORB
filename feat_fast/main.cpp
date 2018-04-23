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
#include "opencvTest.hpp"
#include "ORB.hpp"

//====================================================================================
int main(int argc, char const *argv[]) {
	//#define harrisTest
#ifdef harrisTest
	opencvHarris3();
#else
	string name1 = "srcImg\\sc02.bmp", name2= "srcImg\\sc03.bmp";
	//string name1 = "srcImg\\ball_01.bmp", name2= "srcImg\\ball_02.bmp";
	// 開圖
	ImgRaw img1(name1, "", 0);
	ImgRaw img1_gray = img1.ConverGray();
	img1_gray.nomal=0;
	basic_ImgData warpL, warpR, lapblend;
	ImgRaw imgL(name1);
	ImgRaw imgR(name2);
	// 開圖
	ImgRaw img2(name2, "", 0);
	ImgRaw img2_gray = img2.ConverGray();
	img2_gray.nomal=0;
	ImgData_read(warpL, name1);
	ImgData_read(warpR, name2);


	//====================================================================================
	Timer total, t1;
	t1.priSta=1;
	// ORB
	Feat feat, feat2;
	t1.start();
	create_ORB(img1_gray, feat);
	t1.print(" create_ORB1"); // 0.274
	cout << "===============" << endl;
	t1.start();
	create_ORB(img2_gray, feat2);
	t1.print(" create_ORB2"); // 0.274
	// 尋找配對點
	vector<float> HomogMat;
	t1.start();
	matchORB(feat2, feat, HomogMat);
	t1.print(" matchORB"); // 0.006
	// 測試配對點
	//ImgRaw stackImg = imgMerge(img1, img2);
	//stackImg.bmp("merge.bmp");
	//featDrawLine("line.bmp", stackImg, feat2);

	cout << "=======================================" << endl;
	// 縫合圖片
	size_t RANSAC_num=0;
	Feature** RANSAC_feat=nullptr;
	//RANSAC_feat = new Feature*[RANSAC_num];
	getNewfeat(feat2, RANSAC_feat, RANSAC_num);
	//featDrawLine2("_matchImg_RANSACImg.bmp", stackImg, RANSAC_feat, RANSAC_num);
	//====================================================================================

	// 獲得偏差值
	int mx=0, my=0; float ft=0;
	t1.start();
	ft = getWarpFocal(HomogMat, imgL.size(), imgR.size());
	t1.print(" getWarpFocal"); // 0
	t1.start();
	getWarpOffset(imgL, imgR, RANSAC_feat, RANSAC_num, mx, my, ft);
	t1.print(" getWarpOffset"); // 0.13
	//cout << "ft=" << ft << ", Offset(" << mx << ", " << my << ")" << endl;
	//====================================================================================
	t1.start();
	LapBlender(lapblend, warpL, warpR, ft, mx, my);
	t1.print(" LapBlender"); // 0.022
	cout << "=======================================" << endl;
	total.print("#### total time"); //0.093
	ImgData_write(lapblend, "__lapBlend.bmp");

#endif // harrisTest
	return 0;
}
//====================================================================================