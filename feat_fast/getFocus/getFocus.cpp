/*****************************************************************
Name : imgraw
Date : 2018/04/20
By   : CharlotteHonG
Final: 2018/04/20
*****************************************************************/
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

#include "Timer.hpp"
#include "Imgraw.hpp"
#include "getFocus.hpp"






// 輸入 仿射矩陣 獲得焦距
static void focalsFromHomography(const vector<float> &HomogMat, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
	const auto& h = HomogMat;

	double d1, d2; // Denominators
	double v1, v2; // Focal squares value candidates

	f1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f1 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = std::sqrt(v1);
	else f1_ok = false;

	f0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f0 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f0 = std::sqrt(v1);
	else f0_ok = false;
}
// 獲得焦距(所有圖共用一個ft).
double getWarpFocal(const vector<float> &HomogMat, size_t img1Size, size_t img2Size) {
	int img_total = 2;
	double f0 = 0.f, f1 = 0.f, ft = 0.f;
	bool f0ok = false, f1ok = false;

	vector<float> all_focals;
	if(!HomogMat.empty()) {
		focalsFromHomography(HomogMat, f0, f1, f0ok, f1ok);
		if(f0ok && f1ok) {
			all_focals.push_back(sqrtf(f0 * f1));
		}
	}
	if(all_focals.size() >= img_total - 1) {
		sort(all_focals.begin(), all_focals.end());
		if(all_focals.size() % 2 == 1) {
			ft = all_focals[all_focals.size() / 2];
		} else {
			ft = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5f;
		}
	} else {
		float focals_sum = 0.f;
		focals_sum += img1Size + img2Size;
		ft = focals_sum / (float)img_total;
	}
	//cout << "ft = " << ft << endl;
	return ft;
}




// 對齊取得第二張圖偏移量
void getWarpOffset(const ImgRaw &imgA, const ImgRaw &imgB,
	Feature const* const* good_match, int gm_num,
	int &x, int &y, float FL)
{
	//------------------------------------------------------------------------
	// 轉換用函式.
	auto&& raw_to_imgraw = [](const Raw& src){
		ImgRaw dst(src.RGB, src.getCol(), src.getRow(), 24);
		return dst;
	};
	auto&& imgraw_to_raw = [](const ImgRaw& src){
		Raw dst(src.width, src.height);
		dst.RGB = src; // 這裡會呼叫重載函式轉uch
		return dst;
	};
	// 宣告所需資料項目.
	vector<Raw> InputImage = {
		imgraw_to_raw(imgA),
		imgraw_to_raw(imgB)
	};
	//------------------------------------------------------------------------
	Timer t1;
	// todo 這裡就浪費6毫秒
	Raw img1=imgraw_to_raw(imgA), img2=imgraw_to_raw(imgB);
	//t1.print("這裡就浪費6毫秒");

	int cal_dx = 0;
	int cal_dy = 0;
	// 中間值.
	const float&& mid_x1 = (float)img1.getCol() / 2.f;
	const float&& mid_x2 = (float)img2.getCol() / 2.f;
	const float&& mid_y1 = (float)img1.getRow() / 2.f;
	const float&& mid_y2 = (float)img2.getRow() / 2.f;
	// 先算平方.
	const float& fL1 = FL;
	const float& fL2 = FL;
	const float&& fL1_pow = pow(fL1, 2);
	const float&& fL2_pow = pow(fL2, 2);

	Timer t;
//#pragma omp parallel for
	for (int i = 0; i < gm_num-1; i++) {
		Feature const* const& curr_m = good_match[i];
		const float imgX1 = curr_m->x;
		const float imgY1 = curr_m->y;
		const float imgX2 = curr_m->fwd_match->x;
		const float imgY2 = curr_m->fwd_match->y;
		// 圖1
		float theta1 = fastAtanf_rad((imgX2 - mid_x1) / fL1);
		float h1 = imgY2 - mid_y1;
		h1 /= sqrt(pow((imgX2 - mid_x1), 2) + fL1_pow);
		int x1 = (int)(fL1*theta1 + mid_x1+.5);
		int y1 = (int)(fL1*h1 + mid_y1+.5);
		// 圖2
		float theta2 = fastAtanf_rad((imgX1 - mid_x2) / fL2);
		float h2 = imgY1 - mid_y2;
		h2 /= sqrt(pow((imgX1 - mid_x2), 2) + fL2_pow);
		int x2 = (int)(fL2*theta2 + mid_x2 + img1.getCol() +.5);
		int y2 = (int)(fL2*h2 + mid_y2 +.5);
		// 累加座標.
		int distX = x2 - x1;
		int distY = img1.getRow() - y1 + y2;
		cal_dx += distX;
		cal_dy += distY;
	}
	//t.print("match time.");

	// 平均座標.
	int avg_dx = (float)cal_dx / (float)(gm_num-1);
	int avg_dy = (float)cal_dy / (float)(gm_num-1);

	// 修正座標(猜測是4捨5入哪裡怎樣沒寫好才變成這樣).
	if(avg_dx % 2 == 0){
		if(avg_dx + 1 <= img1.getCol() && avg_dx + 1 <= img2.getCol()){
			avg_dx += 1;
		} else{
			avg_dx -= 1;
		}
	}
	if(avg_dy % 2 == 0){
		if(avg_dy + 1 <= img1.getRow() && avg_dy + 1 <= img2.getRow()){
			avg_dy += 1;
			//cout << "		############ this is up" << endl;

		} else{
			avg_dy -= 1;
			//cout << "		############ this is dw" << endl;

		}
	} else if(avg_dy % 2 == 1){
		avg_dy+=1;
		//cout << "		############ this is else" << endl;
	}

	// 輸出座標.
	x=(avg_dx);
	y=(avg_dy);

	int xMove , yMove;
	int xM, yM;

	// 假如 y 的偏移量大於圖片高
	if(y > imgA.height) {
		int dyy = -((int)imgA.height - abs((int)imgA.height - y));
		xMove = x;
		yMove = dyy;

		xM = imgA.width - xMove;
		yM = -(int)imgA.height - yMove;
	} else { // 通常情況
		xMove = x;
		yMove = y;

		xM = imgA.width - xMove;
		yM = imgA.height - yMove;
	}
	//cout << "(x, y)Move = " << xM << ", " << yM << endl;
	x=xM;
	y=yM;
}




