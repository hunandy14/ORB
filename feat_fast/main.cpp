#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace std;

#include "Raw2Img\Raw2img.hpp"
#include "Imgraw\Imgraw.hpp"

extern "C" {
#include "fastlib\fast.h"
}

#include <opencv2\opencv.hpp> 
using namespace cv;

#include "harris_coners.hpp"

#define Fast_THRE 10

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

class Feat{
public:
	Feat(): feat(nullptr), size(0){}
	~Feat(){
		if(feat){
			free(feat);
			feat = nullptr;
			size = 0;
		}
	}
public:
	void fast(ImgRaw& img){
		vector<unsigned char> raw_data = img;
		uchar* data = raw_data.data();
		const int xsize = img.width, ysize = img.height, stride = xsize, threshold = 16;
		this->feat = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &size);
	}
	void harris(ImgRaw& img){
		xy* feat_harris = nullptr;
		int numcorners_harris = 0;
		harris_coners(img, feat_harris, &numcorners_harris, feat, size);
		feat = feat_harris;
		size = size;
	}
public:
	xy& operator[](size_t idx) { return feat[idx]; }
	const xy& operator[](size_t idx) const { return feat[idx]; }

	operator xy* (){
		return feat;
	}
public:
	xy* feat = nullptr;
	int size = 0;
};
void outFeat2bmp(string name, const ImgRaw& img, const Feat& feat){
	ImgRaw temp = img;
	for(size_t i = 0; i < feat.size; i++){
		int x = feat[i].x;
		int y = feat[i].y;
		temp.at2d(y, x) = 0;
	}
	temp.bmp(name);
}
//====================================================================================
void centerOfMass(){
	float v[9]={
		1, 2, 3,
		4, 5, 6,
		7, 8, 9
	};
	int radius = 1;
	float m01=0, m10=0;
	float* center= v+5;
	for(int j = -radius; j <= radius; j++){
		for(int i = -radius; i <= radius; i++){
			//int thisx = feat_point[point_nub].x + i;
			int thisx = 1 + i;
			//int thisx = feat_point[point_nub].x + i;
			int thisy = 1 + j;
			//const byte* p = im + thisy * stride + thisx;
			float* p = center + thisy*3 + thisx;

			cout << *p << ", ";
			//m10
			//if(thisx < 0.0 || thisx >= xsize){
				//cout << "特徵點太過靠近邊緣" << endl;
			//} else{
				m10 += i * (*p);
			//}
			//m01
			//if(thisy + j < 0.0 || thisy + j >= ysize){
				//cout << "特徵點太過靠近邊緣" << endl;
			//} else{
				m01 += j * (*p);
			//}
		}
	}
	cout << "m01=" << m01 << endl;
	cout << "m10=" << m10 << endl;
	float ang = fastAtan2(m01, m10);
	cout << "ang=" << ang << endl;
}

vector<double> GWCM(ImgRaw& img,
	xy* feat_point, int ret_num_corners, int radius)
{
	const float* im=img.raw_img.data();
	int xsize=img.width;
	int ysize=img.height;
	int stride=img.width;

	/*xy* feat_point;
	int ret_num_corners;
	int radius;*/


	int point_nub = 0;
	vector<double> feat_sita(ret_num_corners);

	while (point_nub < ret_num_corners)
	{
		double m01 = 0;
		double m10 = 0;
		for (int j = -radius; j <= radius; j++)
		{
			for (int i = -radius; i <= radius; i++)
			{
				int thisx = feat_point[point_nub].x + i;
				int thisy = feat_point[point_nub].y + j;
				const float* p = im + thisy * stride + thisx;
				int posi = thisy * stride + thisx;
				//m10
				if (thisx < 0.0 || thisx >= xsize) {

					//cout << "特徵點太過靠近邊緣" << endl;
				}
				else
				{
					// todo 這裡不知道為什麼存取都有問題，我有修過imraw
					cout << "pos" << posi << endl;
					double iii = i*img[posi];
					m10 += iii;
				}
				//m01
				if (thisy < 0.0 || thisy >= ysize) {

					//cout << "特徵點太過靠近邊緣" << endl;
				}
				else
				{
					m01 += j*img[posi];
				}
			}
		}
		feat_sita[point_nub++] = fastAtan2(m01, m10);
	}
	return feat_sita;
}
int main(int argc, char const *argv[]){
	// 讀寫 Bmp
	ImgRaw img1("kanna30.bmp");
	img1 = img1.ConverGray();
	// FAST 找特徵點.
	Feat feat;
	feat.fast(img1);
	outFeat2bmp("FAST.bmp", img1, feat);

	int xsize = img1.width, ysize = img1.height, stride = xsize, threshold = 10, numcorners=feat.size;
	// Harris 角點偵測.
	feat.harris(img1);
	outFeat2bmp("harris.bmp", img1, feat);


	vector<unsigned char> raw_data = img1;
	uchar* data = raw_data.data();
	vector<double> sing = 
		//GWCM(data, xsize, ysize, stride, feat, feat.size, 3);
	GWCM(img1, feat, feat.size, 3);

	ImgRaw temp = img1;
	for(size_t i = 0; i < feat.size; i++){
		Draw::draw_arrow(temp, feat[i].y, feat[i].x, 20, sing[i]);
	}
	temp.bmp("arrow.bmp");

	return 0;
}




//====================================================================================
//灰度重心法.
vector<double> GWCM(const byte* im, int xsize, int ysize, int stride, 
	xy* feat_point, int ret_num_corners, int radius)
{
	int point_nub = 0;
	long long m10, m01;
	vector<double> feat_sita(ret_num_corners);


	while (point_nub < ret_num_corners)
	{
		m01 = 0;
		m10 = 0;
		for (int j = -radius; j <= radius; j++)
		{
			for (int i = -radius; i <= radius; i++)
			{
				int thisx = feat_point[point_nub].x + i;
				int thisy = feat_point[point_nub].y + j;
				const byte* p = im + thisy * stride + thisx;
				//m10
				if (thisx < 0.0 || thisx >= xsize) {

					//cout << "特徵點太過靠近邊緣" << endl;
				}
				else
				{
					m10 += (i * (*p));
				}
				//m01
				if (thisy < 0.0 || thisy >= ysize) {

					//cout << "特徵點太過靠近邊緣" << endl;
				}
				else
				{
					m01 += j * (long)(*p);
				}
			}
		}
		feat_sita[point_nub++] = fastAtan2(m01, m10);
	}
	return feat_sita;
}