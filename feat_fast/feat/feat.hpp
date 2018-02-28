#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "harris_coners.hpp"

extern "C" {
#include "fastlib\fast.h"
}

using namespace std;
typedef unsigned char uchar;


class Feat{
public:
	Feat(): feat(nullptr), len(0){}
	~Feat(){
		if(feat){
			free(feat);
			feat = nullptr;
			len = 0;
		}
	}
public:
	void fast(const ImgRaw& img){
		vector<unsigned char> raw_data = img;
		uchar* data = raw_data.data();
		const int xsize = img.width, ysize = img.height, stride = xsize, threshold = 16;
		this->feat = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &len);
	}
	void harris(const ImgRaw& img){
		harris_coners(img, &feat, &len);
	}
public:
	xy & operator[](size_t idx){ return feat[idx]; }
	const xy& operator[](size_t idx) const{ return feat[idx]; }

	operator xy* (){
		return feat;
	}

	const int size() const{
		return len;
	}
public:
	xy* feat = nullptr;
private:
	int len = 0;
};