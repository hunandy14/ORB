#pragma once

#include "ImgRaw\Imgraw.hpp"

extern "C" {
#include "fastlib\fast.h"
}
#include "feat\feat.hpp"

class Coor{
private:
	struct xy{int x, y;}; 
public:
	Coor(size_t num): c(num){}
public:
	vector<xy> c;
};

void harris_coners(const ImgRaw& img, Feat& feat);