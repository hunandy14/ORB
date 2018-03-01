#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace std;
#include <opencv2\opencv.hpp> 
using namespace cv;
//====================================================================================
#include "Imgraw.hpp"
#include "feat\feat.hpp"
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

double average(const ImgRaw& img, int x, int y) {
	double avg = 0.0;
	for(int j = -2; j <= 2; j++) {
		for(int i = -2; i <= 2; i++) {
			int thisx = x + i;
			int thisy = y + j;
			int posi = thisy * img.width + thisx;
			if((thisx < 0.0 or thisx >= img.width) or
				(thisy < 0.0 or thisy >= img.height)) {
				throw out_of_range("出現負號");
			} else {
				avg += img[posi];
			}
		}
	}
	avg /= 25;
	return avg;
}

bool Compare(const ImgRaw& img, Feat& feat, int x1, int y1, int x2, int y2) {
	double point1, point2;
	point1 = average(img, x1, y1);
	point2 = average(img, x2, y2);
	if(point1 > point2) return true;
	else return false;
}
// n is mask size
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



	return 0;
}




//====================================================================================