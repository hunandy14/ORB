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

// 漢明距離
void create_ORB(const ImgRaw & img, Feat & feat);
// 配對ORB
void matchORB(Feat & feat1, const Feat & feat2, vector<double>& HomogMat);
// 合併兩張圖
ImgRaw imgMerge(const ImgRaw & img1, const ImgRaw & img2);
// 畫線 (對本地資料結構)
void featDrawLine(string name, const ImgRaw & stackImg, const Feat & feat);
// 畫線 (對blend資料結構)
void featDrawLine2(string name, const ImgRaw & stackImg, Feature const * const * RANfeat, size_t RANfeatNum);
// 轉換到 blen 的資料結構
void getNewfeat(const Feat & feat, Feature **& RANfeat, size_t & RANfeatNum);
