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


void imgStitch(string name1, string name2, string outName="__lapBlend.bmp", bool autoname=0) {
	// 開圖
	ImgRaw img1(name1, "", 0);
	ImgRaw img1_gray = img1.ConverGray();
	img1_gray.nomal=0;
	ImgRaw img2(name2, "", 0);
	ImgRaw img2_gray = img2.ConverGray();
	img2_gray.nomal=0;

	ImgRaw stackImg = imgMerge(img1, img2);

	ImgRaw imgL(name1);
	ImgRaw imgR(name2);

	basic_ImgData warpL, warpR, lapblend;
	ImgData_read(warpL, name1);
	ImgData_read(warpR, name2);
	//====================================================================================
	Timer t1;
	Timer total;
	//t1.priSta=1;
	// ORB
	Feat feat, feat2;
	t1.start();
	create_ORB(img1_gray, feat); // 0.25ms
	t1.print(" create_ORB1");

	//t1.start();
	create_ORB(img2_gray, feat2); // 0.25ms
	//t1.print(" create_ORB2"); 

	// 尋找配對點
	vector<double> HomogMat;
	t1.start();
	matchORB(feat2, feat, HomogMat); // 1ms
	t1.print(" matchORB");

	// 測試配對點
	//stackImg.bmp("merge.bmp");
	//featDrawLine("line.bmp", stackImg, feat2);

	// 縫合圖片
	size_t RANSAC_num=0;
	Feature** RANSAC_feat=nullptr;
	//RANSAC_feat = new Feature*[RANSAC_num];
	getNewfeat(feat2, RANSAC_feat, RANSAC_num);
	//featDrawLine2("_matchImg_RANSACImg.bmp", stackImg, RANSAC_feat, RANSAC_num);



	//====================================================================================
	// 獲得偏差值
	int mx, my; double focals;
	//t1.start();
	estimateFocal(HomogMat, focals); // 0ms
	//t1.print(" getWarpFocal");
	//t1.start();
	getWarpOffset(imgL, imgR, RANSAC_feat, RANSAC_num, mx, my, focals); // 0ms
	//t1.print(" getWarpOffset");
	cout << "ft=" << focals << ", Ax=" << mx << ", Ay=" << my << ";" << endl;



	//====================================================================================
	//t1.start();
	LapBlender(lapblend, warpL, warpR, focals, mx, my); // 22ms
	//WarpPers_Stitch(lapblend, warpL, warpR, HomogMat);
	//t1.print(" LapBlender");
	//cout << "=======================================" << endl;
	total.print("# total time"); // 93ms

	
	static int num=0;
	if (autoname) {
		outName = outName+to_string(num++)+".bmp";
		cout << outName << endl;
		ImgData_write(lapblend, outName);
	}
	else {
		ImgData_write(lapblend, outName);
	}
}
//====================================================================================
int main(int argc, char const *argv[]) {
	imgStitch("srcImg\\sc02.bmp", "srcImg\\sc03.bmp", "resultImg\\sc02_blend.bmp");
	//imgStitch("srcImg\\ball_01.bmp", "srcImg\\ball_02.bmp", "resultImg\\ball_01_blend.bmp");

	imgStitch("data\\DSC_2936.bmp", "data\\DSC_2937.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2938.bmp", "data\\DSC_2939.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2940.bmp", "data\\DSC_2941.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2942.bmp", "data\\DSC_2943.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2944.bmp", "data\\DSC_2945.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2946.bmp", "data\\DSC_2947.bmp", "resultImg\\blend", 1);
	imgStitch("data\\DSC_2950.bmp", "data\\DSC_2951.bmp", "resultImg\\blend", 1);
	return 0;
}
//====================================================================================