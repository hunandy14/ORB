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
#include "opencvTest.hpp"
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
					cout << "thisx=" << thisx << endl;
					cout << "thisy=" << thisy << endl;
					//throw out_of_range("出現負號");
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
	const int r = 2;
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
	avg /= 25.0;
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
		
		// 描述點對
		const int singIdx = floor(sing[k]/30.f);
		int x1 = x + bit_pattern_31[singIdx][k*4 + 0];
		int y1 = y + bit_pattern_31[singIdx][k*4 + 1];
		int x2 = x + bit_pattern_31[singIdx][k*4 + 3];
		int y2 = y + bit_pattern_31[singIdx][k*4 + 4];

		// 不旋轉
		/*const int singIdx = 0;
		int x1 = x + bit_pattern_31[singIdx][k*4 + 0];
		int y1 = y + bit_pattern_31[singIdx][k*4 + 1];
		int x2 = x + bit_pattern_31[singIdx][k*4 + 3];
		int y2 = y + bit_pattern_31[singIdx][k*4 + 4];*/
		
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
void create_ORB(const ImgRaw& img, Feat& feat) {
	// FAST特徵點
	fast(img, feat);
	// Harris過濾角點
	//harris_coners(img, feat);


	Mat image(img.height, img.width, CV_32F, (void*)img.raw_img.data());
	Mat gray=image;
	// 初始化mask
	Mat mask(Mat::zeros(Size(img.width, img.height),CV_8U));
	Mat mask2(Mat::ones(Size(img.width, img.height),CV_8U));
	// 把 feat 的 xy 轉到 mask
	int edg=30;
	for(size_t i = 0; i < feat.len; i++) {
		//idx = (feat.feat->y)*image.rows + (feat.feat->x);
		int x=feat[i].x;
		int y=feat[i].y;
		Point pt(x, y);
		//cout << "string=" << pt << endl;

		// 過濾邊緣位置
		if(x>=(3+edg) and x<=img.width-(3+edg) && y>=(3+edg) and y<=img.height-(3+edg)) {
			mask.at<uchar>(pt) = 255;
		}
	}
	
	//cout << "idx=" << feat.len << endl;
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, 1000, 0.01, 10, mask, 3, true, 0.04);
	cout << "corners=" << corners.size() << endl;


	// 回填 xy 位置
	int newLen=0;
	for(int i = 0; i < corners.size(); i++) {
		int x=corners[i].x;
		int y=corners[i].y;
		
		if(x<edg) {
			throw("x<edg");
		}
		if(i==1) {
			cout << "x=" << x << endl;
		}

		feat[i].x = x;
		feat[i].y = y;
		newLen++;
	}
	feat.len=corners.size();

	/*
	RNG rng(12345);
	image = imread("sc02.bmp");
	for(size_t i = 0; i < corners.size(); i++){
		Scalar color;
		color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		color = Scalar(0, 0, 255);
		circle(image, corners[i], 5, color, 1);
	}
	imshow("goodFeaturesToTrack", image);
	waitKey();
	*/




	// 灰度重心法
	GrayCenterOfMass(img, feat, 3);
	// 描述特徵
	desc_ORB(img, feat);
}

// 漢明距離
int hamDist(const OrbDest& a, const OrbDest& b) {
	return (a^b).count();
}
// 配對ORB
void matchORB(Feat& feat1, const Feat& feat2) {
	// todo 這裡還沒 delete
	feat1.feat_match = new xy[feat1.size()];

	for(size_t j = 0; j < feat1.size(); j++) {
		int dist = numeric_limits<int>::max();
		int matchIdx = -1;
		int distCurr = 0;
		for(size_t i = 0; i < feat2.size(); i++) {
			distCurr = hamDist(feat1.bin[j], feat2.bin[i]);
			// 距離較短則更新
			if(distCurr < dist) {
				dist = distCurr;
				matchIdx = i;
			}
		}
		// 加入匹配點
		//cout << "distCurr=" << distCurr << endl;
		if(distCurr > 128 or
			abs(feat1.feat[j].y - feat2.feat[matchIdx].y) > 1000 )
		{
			feat1.feat_match[j].x = -1;
			feat1.feat_match[j].y = -1;
		} else {
			feat1.feat_match[j].x = feat2.feat[matchIdx].x;
			feat1.feat_match[j].y = feat2.feat[matchIdx].y;
		}
	}
}

// 合併兩張圖
ImgRaw imgMerge(const ImgRaw& img1, const ImgRaw& img2) {
	ImgRaw stackImg;
	int Width  = img1.width+img2.width;
	int Height = img1.height;
	// 合併兩張圖
	stackImg.resize(img1.width*2, img2.height, 24);
	for (size_t j = 0; j < img1.height; j++) {
		for (size_t i = 0; i < img1.width; i++) {
			stackImg[(j*stackImg.width+i)*3 + 0] = img1[(j*img1.width+i)*3 + 0];
			stackImg[(j*stackImg.width+i)*3 + 1] = img1[(j*img1.width+i)*3 + 1];
			stackImg[(j*stackImg.width+i)*3 + 2] = img1[(j*img1.width+i)*3 + 2];
		}
		for (size_t i = img1.width; i < img2.width+img1.width; i++) {
			stackImg[(j*stackImg.width+i)*3 + 0] = img2[(j*img1.width+i-img1.width)*3 + 0];
			stackImg[(j*stackImg.width+i)*3 + 1] = img2[(j*img1.width+i-img1.width)*3 + 1];
			stackImg[(j*stackImg.width+i)*3 + 2] = img2[(j*img1.width+i-img1.width)*3 + 2];
		}
	}
	return stackImg;
}
static void featDrawLine(string name, const ImgRaw& stackImg, const Feat& feat) {
	size_t featNum = feat.size();
	ImgRaw outImg = stackImg;
	for(int i = 0; i < featNum; i++) {
		if(feat.feat_match[i].x != -1) {
			const int& x1 = feat.feat[i].x;
			const int& y1 = feat.feat[i].y;
			const int& x2 = feat.feat_match[i].x + (outImg.width *.5);
			const int& y2 = feat.feat_match[i].y;
			/*if(x1 < 100) {
				cout << "size=" <<feat.size() << ", i=" << i << ", x1="<<x1<<endl;
				throw ("X>100");
			}*/
			Draw::drawLineRGB_p(outImg, y1, x1, y2, x2);
		}
	}
	outImg.bmp(name, 24);
}
//====================================================================================
int main(int argc, char const *argv[]) {
//#define harrisTest
#ifdef harrisTest
	opencvHarris3();
#else

	// 開圖
	ImgRaw img1("sc02.bmp");
	ImgRaw img1_gray = img1.ConverGray();
	// ORB
	Feat feat;
	create_ORB(img1_gray, feat);

	
	// 開圖
	ImgRaw img2("sc03.bmp");
	ImgRaw img2_gray = img2.ConverGray();
	// ORB
	Feat feat2;
	create_ORB(img2_gray, feat2);

	
	// 尋找配對點
	matchORB(feat, feat2);
	
	// 測試配對點
	ImgRaw stackImg = imgMerge(img1, img2);
	stackImg.bmp("merge.bmp");
	featDrawLine("line.bmp", stackImg, feat);
	
	
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