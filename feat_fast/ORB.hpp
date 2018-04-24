/*****************************************************************
Name : ORB
Date : 2018/03/01
By   : CharlotteHonG
Final: 2018/03/19
*****************************************************************/
#pragma once
#include "Imgraw.hpp"
#include "feat.hpp"
#include "getFocus\getFocus.hpp"
#include "LapBlend\LapBlend.hpp"
#include "WarpPers\WarpPers.hpp"

// �~���Z��
void create_ORB(const ImgRaw & img, Feat & feat);
// �t��ORB
void matchORB(Feat & feat1, const Feat & feat2, vector<double>& HomogMat);
// �X�֨�i��
ImgRaw imgMerge(const ImgRaw & img1, const ImgRaw & img2);
// �e�u (�糧�a��Ƶ��c)
void featDrawLine(string name, const ImgRaw & stackImg, const Feat & feat);
// �e�u (��blend��Ƶ��c)
void featDrawLine2(string name, const ImgRaw & stackImg, Feature const * const * RANfeat, size_t RANfeatNum);
// �ഫ�� blen ����Ƶ��c
void getNewfeat(const Feat & feat, Feature **& RANfeat, size_t & RANfeatNum);
