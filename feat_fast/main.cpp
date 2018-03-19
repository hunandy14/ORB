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
#include "harris_coners.hpp"
#include "ORB_bit_pattern_31_.hpp"
#include "opencvTest.hpp"


#include "stitch\imagedata.hpp"
#include "stitch\Blend.hpp"

int main(int argc, char const *argv[]) {
//#define harrisTest
#ifdef harrisTest
	opencvHarris3();
#else
	string name1 = "sc02.bmp", name2= "sc03.bmp";
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


void get_ORB_bin_test() {
	// 讀寫 Bmp
	ImgRaw img1("kanna.bmp");
	img1 = img1.ConverGray();
	// FAST 找特徵點.
	Feat feat;
	feat.fast(img1);
	outFeat2bmp("FAST.bmp", img1, feat);
	// Harris 角點偵測.
	harris_coners(img1, feat);
	outFeat2bmp("harris.bmp", img1, feat);
	// 灰度質心
	GrayCenterOfMass(img1, feat, 3);
	vector<double> sing = feat.sita;
	// 畫箭頭
	ImgRaw temp = img1;
	for(size_t i = 0; i < feat.size(); i++) {
		Draw::draw_arrow(temp, feat[i].y, feat[i].x, 20, sing[i]);
	}
	temp.bmp("arrow.bmp");
	// 模糊
	ImgRaw img2;
	Lowpass(img1, img2, 3);
	img2.bmp("Lowpass.bmp");

	// 描述
	vector<OrbDest> bin(feat.size());
	for(size_t i = 0; i < feat.size(); i++) {
		// 描述
		int x = feat[i].x;
		int y = feat[i].y;
		bin[i] = descriptor_ORB(img2, x, y, sing[i]);
	}

	// 測試
	for(size_t i = 0; i < 2; i++) {
		for(size_t k = 0; k < 128; k++) {
			cout << bin[i][k];
		} cout << endl;
	}
	int dis = hamDist(bin[0], bin[1]);
	cout << "dis=" << dis << endl;


}