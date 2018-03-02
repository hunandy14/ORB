﻿#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>
using namespace std;
#include <opencv2\opencv.hpp> 
using namespace cv;
//====================================================================================
#include "Imgraw.hpp"
#include "feat.hpp"
#include "harris_coners.hpp"
#include "ORB_bit_pattern_31_.hpp"

//====================================================================================
void outFeat2bmp(string name, const ImgRaw& img, const Feat& feat) {
	ImgRaw temp = img;
	for(size_t i = 0; i < feat.size(); i++) {
		int x = feat[i].x;
		int y = feat[i].y;
		temp.at2d(y, x) = 0;
	}
	temp.bmp(name);
}
vector<double> GWCM(const ImgRaw& img, Feat& feat, int radius) {
	vector<double> feat_sita(feat.size());
	for(int idx = 0; idx < feat.size();) {
		double m01 = 0;
		double m10 = 0;
		for(int j = -radius; j <= radius; j++) {
			for(int i = -radius; i <= radius; i++) {
				int thisx = feat[idx].x + i;
				int thisy = feat[idx].y + j;
				int posi = thisy * img.width + thisx;
				if((thisx < 0.0 or thisx >= img.width) or
					(thisy < 0.0 or thisy >= img.height)) {
					throw out_of_range("出現負號");
				} else {
					m10 += i * img[posi];
					m01 += j * img[posi];
				}
			}
		}
		feat_sita[idx++] = fastAtan2(m01, m10);
	}
	return feat_sita;
}
//====================================================================================
// 積分模糊
void Lowpass(const ImgRaw& img, ImgRaw& newimg, int n = 3) {
	newimg = img;
	if(n < 3) {
		throw out_of_range("遮罩不該小於3");
	}
	int dn = (n - 1) / 2;
	for(int j = dn; j < img.height - dn; j++) {
		for(int i = dn; i < img.width - dn; i++) {
			int total = 0;
			int thisx = i;
			int thisy = j;
			int posi = thisy * img.width + thisx;
			for(int dy = -dn; dy <= dn; dy++) {
				for(int dx = -dn; dx <= dn; dx++) {
					thisx = i + dx;
					thisy = j + dy;
					posi = thisy * img.width + thisx;
					total += img[posi];
				}
			}
			newimg[posi] = (int)(total / 9);
		}
	}
}
// 取平均
static double average(const ImgRaw& img, int x, int y) {
	double avg = 0.0;
	int r = 2;
	for(int j = -r; j <= r; j++) {
		for(int i = -r; i <= r; i++) {
			int thisx = x + i;
			int thisy = y + j;
			// 處理負號
			if(thisx < 0.0 ) {
				thisx = 0;
			}
			if(thisx >= img.width) {
				thisx = img.width-1;
			}
			if(thisy < 0.0) {
				thisy = 0;
			}
			if(thisy >= img.height) {
				thisy = img.height-1;
			}
			// 累加
			int posi = thisy * img.width + thisx;
			avg += img[posi];
		}
	}
	avg /= 25;
	return avg;
}
// 描述特徵點內的一個bit
static bool Compare(const ImgRaw& img, int x1, int y1, int x2, int y2) {
	double point1 = average(img, x1, y1);
	double point2 = average(img, x2, y2);

	if(point1 > point2) {
		return true;
	} else {
		return false;
	}
}
// 描述一個特徵點
using OrbDest = bitset<128>;
OrbDest destp(const ImgRaw& img, int x, int y, vector<double> sing) {
	OrbDest bin;
	for(size_t k = 0; k < 128; k++) {
		// 根據角度選不同位移組
		int singIdx = floor(sing[k]/30.f);
		// 描述點對
		int x1 = x + bit_pattern_31[singIdx][k*4 + 0];
		int y1 = y + bit_pattern_31[singIdx][k*4 + 1];
		int x2 = x + bit_pattern_31[singIdx][k*4 + 3];
		int y2 = y + bit_pattern_31[singIdx][k*4 + 4];
		
		bin[k] = Compare(img, x1, y1, x2, y2);
	}
	return bin;
}
// 漢明距離
int hamDist(const OrbDest& a, const OrbDest& b) {
	return (a^b).count();
}
// 配對ORB
void matchORB(Feat& feat1, const Feat& feat2) {

}
//====================================================================================
int main(int argc, char const *argv[]) {
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
	vector<double> sing = GWCM(img1, feat, 3);
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
		bin[i] = destp(img2, x, y, sing);
	}

	// 測試
	for(size_t i = 0; i < 2; i++) {
		for(size_t k = 0; k < 128; k++) {
			cout << bin[i][k];
		} cout << endl;
	}
	int dis = hamDist(bin[0], bin[1]);
	cout << "dis=" << dis << endl;

	// 尋找配對點

	return 0;
}
//====================================================================================