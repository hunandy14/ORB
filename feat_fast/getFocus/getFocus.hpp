/*****************************************************************
Name : imgraw
Date : 2018/04/20
By   : CharlotteHonG
Final: 2018/04/20
*****************************************************************/
#pragma once

#include "stitch\imagedata.hpp"

float getWarpFocal(const vector<float>& HomogMat, 
	size_t img1Size, size_t img2Size);

void getWarpOffset(const ImgRaw &imgA, const ImgRaw &imgB,
	Feature const* const* good_match, int gm_num,
	int &x, int &y, float FL);

