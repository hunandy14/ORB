/*****************************************************************
Name : ORB
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

#include "ORB.hpp"

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
static void GrayCenterOfMass(const ImgRaw& img, Feat& feat, int radius) {
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
		double s = fastAtan2(m01, m10);
		//cout << s << "==> \t" << s/30.0 << "  ==>\t  " << round(s/30.0) << endl;
		//s=round(s/30.0);
		feat_sita[idx++] = s;
	}
	feat.sita = std::move(feat_sita);
}
//====================================================================================
// 獲得FAST特徵點
static void fast(const ImgRaw& img, Feat& feat){
	vector<unsigned char> raw_data = img;
	uchar* data = raw_data.data();
	const int xsize = img.width, ysize = img.height, stride = xsize, threshold = 16;
	feat.feat = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &feat.len);
}
// 積分模糊
static void Lowpass(const ImgRaw& img, ImgRaw& newimg, int n = 3) {
	newimg = img;
	if(n < 3) {
		throw out_of_range("遮罩不該小於3");
	}
	int dn = (n - 1) / 2;

	int i, j, dy;
#pragma omp parallel for private(i, j, dy)
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
			if(thisx < 0.0 or thisx >= img.width-1 or 
				thisy < 0.0 or thisy > img.height-1)
			{
				throw out_of_range("out");
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
static OrbDest descriptor_ORB(const ImgRaw& img, int x, int y, double sing) {
	OrbDest bin;
//#pragma omp parallel for
	for(int k = 0; k < 128; k++) {
		// 根據角度選不同位移組

		// 描述點對
		// todo 乾 這裡好像有錯
		const int singIdx = sing/30.0;
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
static void desc_ORB(const ImgRaw& img, Feat& feat) {
	ImgRaw img2;
	img2.nomal=0;
	Lowpass(img, img2, 3);
	vector<OrbDest> bin(feat.size());
#pragma omp parallel for
	for(int i = 0; i < feat.size(); i++) {
		// 描述
		int x = feat[i].x;
		int y = feat[i].y;
		bin[i] = descriptor_ORB(img2, x, y, (feat.sita)[i]);
	}
	feat.bin = std::move(bin);
}
void create_ORB(const ImgRaw& img, Feat& feat) {
// FAST特徵點 ---> 8~10ms
	fast(img, feat);

	
// 初始化mask --- >0ms
	Mat image(img.height, img.width, CV_32F, (void*)img.raw_img.data());
	Mat gray=image;
	Mat mask(Mat::zeros(Size(img.width, img.height),CV_8U));
	Mat mask2(Mat::ones(Size(img.width, img.height),CV_8U));
	// 把 feat 的 xy 轉到 mask
	int edg=3+20;
	for(int i = 0; i < feat.len; i++) {
		//idx = (feat.feat->y)*image.rows + (feat.feat->x);
		int x=feat[i].x;
		int y=feat[i].y;
		Point pt(x, y);
		//cout << "string=" << pt << endl;

		// 過濾邊緣位置
		if(x>=(edg) and x<=img.width-(edg) && y>=(edg) and y<=img.height-(edg)) {
			mask.at<uchar>(pt) = 255;
		}
	}
	//cout << "FAST corner 數量 = " << feat.len << endl;
	

// Herris 過濾 --> 8~10ms	
	vector<Point2f> corners;
	goodFeaturesToTrack2(gray, corners, 2000, 0.01, 10, mask, 3, 3, true, 0.04); // 8~10ms
	//goodFeaturesToTrack(gray, corners, 1000, 0.01, 10, mask, 3, true, 0.04);
	//cout << "Harris corner 數量 = " << corners.size() << endl;


// 回填 xy 位置 ---> 0ms
	int newLen=0;
	for(int i = 0; i < corners.size(); i++) {
		int x=corners[i].x;
		int y=corners[i].y;
		if(x<edg) {
			throw("x<edg");
		}
		if(i==1) {
			//cout << "x=" << x << endl;
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


// 灰度重心法 ---> 0ms
	GrayCenterOfMass(img, feat, 3);

	/*ImgRaw temp = img;
	for(size_t i = 0; i < feat.size(); i++) {
		Draw::draw_arrow(temp, feat[i].y, feat[i].x, 20, feat.sita[i]);
	}
	static int num=0;
	temp.bmp("arrow"+to_string(num++)+".bmp");*/


// 描述特徵 ---> 5~6ms
	desc_ORB(img, feat);
}

// 漢明距離
static int hamDist(const OrbDest& a, const OrbDest& b) {
	return (a^b).count();
}
// 配對ORB
void matchORB(Feat& feat1, const Feat& feat2, vector<double>& HomogMat) {
	// todo 這裡還沒 delete
	feat1.feat_match = new xy[feat1.size()];

	int max_dist = 0; int min_dist = 100;
	feat1.distance.resize(feat1.size());

	vector<bool> f2_matched(feat2.size());
#pragma omp parallel for
	for(int j = 0; j < feat1.size(); j++) {
		int dist = numeric_limits<int>::max();
		int matchIdx = -1;
		// f1 的這一點與所有 f2 匹配找最短距離
		for(int i = 0; i < feat2.size(); i++) {
			int distCurr = hamDist(feat1.bin[j], feat2.bin[i]);
			// 距離較短則更新
			if(distCurr < dist and f2_matched[i] == 0) {
				dist = distCurr;
				matchIdx = i;
			}
		}
		// 加入匹配點
		//cout << "dist=" << dist << endl;
		if(dist > 24 /*or
					 abs(feat1.feat[j].y - feat2.feat[matchIdx].y) > 1000*/ )
		{
			// 沒配到的標記 -1 待會刪除
			feat1.feat_match[j].x = -1;
			feat1.feat_match[j].y = -1;
		} else {
			// 標記已經配過的f2點
			f2_matched[matchIdx] = 1;
			// 匹配成功
			feat1.feat_match[j].x = feat2.feat[matchIdx].x;
			feat1.feat_match[j].y = feat2.feat[matchIdx].y;
			feat1.distance[j] = dist;
			// 紀錄最大最小距離(做簡單過濾用)
			if(dist!=0 && dist < min_dist) min_dist = dist;
			if(dist > max_dist) max_dist = dist;
		}
	}

	//-- Quick calculation of max and min distances between keypoints
	//printf("-- Max dist : %d \n", max_dist);
	//printf("-- Min dist : %d \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	// 用距離來刪減可能錯誤的點(準度不是很好應該很容易出事)
	// (發現找到的點已經很少了刪減沒什麼必要)
	feat1.distance.resize(feat1.size());
	for(int i = 0; i < feat1.size(); i++){
		if(feat1.distance[i] > 6 * min_dist) {
			cout << "TEST=" << endl;
			//good_matches.push_back(matches[i]);
			feat1.feat_match[i].x = -1;
			feat1.feat_match[i].y = -1;
			feat1.distance[i] = 0;
		}
	}


	// get feat point
	vector<Point2f> featPoint1;
	vector<Point2f> featPoint2;
	for(size_t i = 0; i < feat1.size(); i++){
		//-- Get the keypoints from the good matches
		int x=feat1[i].x;
		int y=feat1[i].y;
		Point pt(x, y);
		int x2=feat1.feat_match[i].x;
		int y2=feat1.feat_match[i].y;
		Point pt2(x2, y2);

		if(x2!=-1 and y2!=-1) {
			featPoint1.push_back(pt);
			featPoint2.push_back(pt2);
		}
	}

	// get Homography and RANSAC mask
	vector<char> RANSAC_mask;
	Mat Hog = findHomography(featPoint1, featPoint2, RANSAC, 3, RANSAC_mask, 2000, 0.995);

	// 更新到 feat
	feat1.len=featPoint1.size();
	for(size_t i = 0; i < featPoint1.size(); i++) {
		if(RANSAC_mask[i]!=0) {
			feat1[i].x = featPoint1[i].x;
			feat1[i].y = featPoint1[i].y;

			feat1.feat_match[i].x = featPoint2[i].x;
			feat1.feat_match[i].y = featPoint2[i].y;
		} else {
			feat1.feat_match[i].x = -1;
			feat1.feat_match[i].y = -1;
		}
	}

	// 輸出到 hog
	HomogMat.resize(Hog.cols*Hog.rows);
	for(size_t j = 0, idx=0; j < Hog.rows; j++) {
		for(size_t i = 0; i < Hog.cols; i++, idx++) {
			HomogMat[idx]=Hog.at<double>(j, i);
			//cout << HomogMat[idx] << ", ";
		} //cout << endl;
	} //cout << endl;

	//cout << "Hog = \n" << Hog << endl;
}


// 合併兩張圖
ImgRaw imgMerge(const ImgRaw& img1, const ImgRaw& img2) {
	ImgRaw stackImg;
	stackImg.nomal=0;
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
// 畫線 (對本地資料結構)
void featDrawLine(string name, const ImgRaw& stackImg, const Feat& feat) {
	size_t featNum = feat.size();
	ImgRaw outImg = stackImg;
	int i=0, idx=0;
	for(i = 0; i < featNum; i++) {
		if(feat.feat_match[i].x != -1) {
			const int& x1 = feat.feat_match[i].x + (outImg.width *.5);
			const int& y1 = feat.feat_match[i].y;
			const int& x2 = feat.feat[i].x;
			const int& y2 = feat.feat[i].y;
			/*if(x1 < 100) {
			cout << "size=" <<feat.size() << ", i=" << i << ", x1="<<x1<<endl;
			throw ("X>100");
			}*/
			Draw::drawLineRGB_p(outImg, y1, x1, y2, x2);
			idx++;
		}
	}
	//cout << "idxxxxxx=" << i << ", " << idx << endl;

	outImg.bmp(name, 24);
}
// 畫線 (對blend資料結構)
void featDrawLine2(string name, const ImgRaw& stackImg, Feature const* const* RANfeat , size_t RANfeatNum) {
	ImgRaw outImg = stackImg;
	int i;
	for(i = 1; i < RANfeatNum; i++) {
		if(RANfeat[i]->fwd_match) {
			/*
			fpoint pt11 = fpoint(round(RANfeat[j]->rX()), round(RANfeat[j]->rY()));
			fpoint pt22 = fpoint(round(RANfeat[j]->fwd_match->rX()), round(RANfeat[j]->fwd_match->rY()));
			*/
			fpoint pt11 = 
				fpoint(round(RANfeat[i]->fwd_match->rX()), round(RANfeat[i]->fwd_match->rY()));
			fpoint pt22 = 
				fpoint(round(RANfeat[i]->rX()), round(RANfeat[i]->rY()));

			const int& x1 = pt11.x;
			const int& y1 = pt11.y;
			const int& x2 = pt22.x + (outImg.width *.5);
			const int& y2 = pt22.y;
			Draw::drawLineRGB_p(outImg, y1, x1, y2, x2);
		} else {
			cerr << "RANSAC feat[i].fwd_match is nullptr" << endl;
		}
	}
	//cout << "idxxxxxx=" << i << endl;
	outImg.bmp(name, 24);
}
// 轉換到 blen 的資料結構
void getNewfeat(const Feat& feat, Feature**& RANfeat , size_t& RANfeatNum) {
	size_t featNum = feat.size();
	RANfeat = new Feature*[featNum]{};

	int i=0, idx=0;
	for(i = 0; i < featNum; i++) {
		if(feat.feat_match[i].x != -1) {
			const int& x1 = 
				feat.feat_match[i].x;
			const int& y1 = 
				feat.feat_match[i].y;
			const int& x2 = 
				feat.feat[i].x;
			const int& y2 = 
				feat.feat[i].y;

			Feature* fm = new Feature{};
			fm->size = 1.0;
			fm->x = x1;
			fm->y = y1;

			Feature* f = new Feature{};
			f->size = 1.0;
			f->x = x2;
			f->y = y2;
			f->fwd_match = fm;

			// 輸入
			RANfeat[idx]=f;
			/*cout << RANfeat[idx]->rX() << ", ";
			cout << RANfeat[idx]->rY() << "---->";
			cout << (RANfeat[idx]->fwd_match)->rX() << ", ";
			cout << (RANfeat[idx]->fwd_match)->rY() << endl;*/

			idx++;
		}
	}
	RANfeatNum=idx;
	//cout << "idxxxxxx=" << i << ", " << idx << endl;
}