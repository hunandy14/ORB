#pragma once
#include <iostream>
#include <vector>
#include <string>

extern "C" {
#include "fastlib\fast.h"
}


using namespace std;
typedef unsigned char uchar;



class Feat{
public:
	Feat(): feat(nullptr), len(0){}
	Feat(int size): feat(nullptr), len(size){
		feat = (xy*)calloc(len, sizeof(xy));
	}
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
public:
	xy & operator[](size_t idx){ return feat[idx]; }
	const xy& operator[](size_t idx) const{ return feat[idx]; }

	operator xy* (){
		return feat;
	}

	const int size() const{
		return len;
	}
	void resize(int size){
		// 建立心空間
		xy* temp = new xy[size];
		// 複製到新空間
		copy_n(feat, size<len? size: len, temp);
		// 取代舊空間
		this->~Feat();
		len = size;
		feat = temp;
	}
public:
	xy* feat = nullptr;
	float sita = 0.0;
public:
	int len = 0;
};