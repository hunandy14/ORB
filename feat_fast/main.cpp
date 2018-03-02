#include <iostream>
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
// 灰度質心法
void GrayCenterOfMass(const ImgRaw& img, Feat& feat, int radius) {
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
	feat.sita = std::move(feat_sita);
}
//====================================================================================
// 獲得FAST特徵點
void fast(const ImgRaw& img, Feat& feat){
	vector<unsigned char> raw_data = img;
	uchar* data = raw_data.data();
	const int xsize = img.width, ysize = img.height, stride = xsize, threshold = 16;
	feat.feat = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &feat.len);
}
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
static OrbDest descriptor_ORB(const ImgRaw& img, int x, int y, vector<double> sing) {
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
// 描述所有特徵點
void desc_ORB(const ImgRaw& img, Feat& feat) {
	ImgRaw img2;
	Lowpass(img, img2, 3);
	vector<OrbDest> bin(feat.size());
	for(size_t i = 0; i < feat.size(); i++) {
		// 描述
		int x = feat[i].x;
		int y = feat[i].y;
		bin[i] = descriptor_ORB(img2, x, y, feat.sita);
	}
	feat.bin = std::move(bin);
}

// 漢明距離
int hamDist(const OrbDest& a, const OrbDest& b) {
	return (a^b).count();
}
// 配對ORB
void matchORB(Feat& feat1, const Feat& feat2) {

}


void create_ORB(const ImgRaw& img, Feat& feat) {
	// FAST特徵點
	fast(img, feat);
	// Harris過濾角點
	harris_coners(img, feat);
	// 灰度重心法
	GrayCenterOfMass(img, feat, 3);
	// 描述特徵
	desc_ORB(img, feat);
}
//====================================================================================
int main(int argc, char const *argv[]) {
	// 讀寫 Bmp
	ImgRaw img1("kanna.bmp");
	img1 = img1.ConverGray();
	// 獲取特徵點.
	Feat feat;
	create_ORB(img1, feat);

	// 測試
	for(size_t i = 0; i < 2; i++) {
		for(size_t k = 0; k < 128; k++) {
			cout << feat.bin[i][k];
		} cout << endl;
	}
	int dis = hamDist(feat.bin[0], feat.bin[1]);
	cout << "dis=" << dis << endl;

	// 尋找配對點

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
		bin[i] = descriptor_ORB(img2, x, y, sing);
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