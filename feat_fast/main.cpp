#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace std;
#include <opencv2\opencv.hpp> 
using namespace cv;
//====================================================================================
#include "Imgraw.hpp"
#include "feat.hpp"


//====================================================================================
void outFeat2bmp(string name, const ImgRaw& img, const Feat& feat){
	ImgRaw temp = img;
	for(size_t i = 0; i < feat.size(); i++){
		int x = feat[i].x;
		int y = feat[i].y;
		temp.at2d(y, x) = 0;
	}
	temp.bmp(name);
}
vector<double> GWCM(const ImgRaw& img, Feat& feat, int radius)
{
	vector<double> feat_sita(feat.size());
	for(int idx = 0; idx < feat.size();){
		double m01 = 0;
		double m10 = 0;
		for(int j = -radius; j <= radius; j++){
			for(int i = -radius; i <= radius; i++){
				int thisx = feat[idx].x + i;
				int thisy = feat[idx].y + j;
				int posi = thisy * img.width + thisx;
				if( (thisx < 0.0 or thisx >= img.width) or
					(thisy < 0.0 or thisy >= img.height) )
				{
					throw out_of_range("出現負號");
				} else{
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
int main(int argc, char const *argv[]){
	// 讀寫 Bmp
	ImgRaw img1("kanna.bmp");
	img1 = img1.ConverGray();
	// FAST 找特徵點.
	Feat feat;
	feat.fast(img1);
	outFeat2bmp("FAST.bmp", img1, feat);

	int xsize = img1.width, ysize = img1.height, stride = xsize, threshold = 10, numcorners = feat.size();
	// Harris 角點偵測.
	feat.harris(img1);
	outFeat2bmp("harris.bmp", img1, feat);


	vector<unsigned char> raw_data = img1;
	uchar* data = raw_data.data();
	vector<double> sing = GWCM(img1, feat, 3);

	ImgRaw temp = img1;
	for(size_t i = 0; i < feat.size(); i++){
		Draw::draw_arrow(temp, feat[i].y, feat[i].x, 20, sing[i]);
	}
	temp.bmp("arrow.bmp");

	return 0;
}




//====================================================================================