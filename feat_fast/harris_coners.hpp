#pragma once

#include "ImgRaw\Imgraw.hpp"
extern "C" {
#include "fastlib\fast.h"
}

class Coor{
private:
	struct xy{int x, y;}; 
public:
	Coor(size_t num): c(num){}
public:
	vector<xy> c;
};

void harris_coners(const ImgRaw & img, xy* & feat_harris, int* numcorners_harris, const xy * feat, int numcorners);