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
static void focalsFromHomography(const vector<double> &HomogMat, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
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
// 從矩陣獲得焦距
double estimateFocal(const vector<double> &HomogMat, size_t img1Size, size_t img2Size) {
	const int num_images = 2;
	double median;

	vector<double> all_focals;
	if(!HomogMat.empty()) {
		double f0 ,f1;
		bool f0ok, f1ok;
		focalsFromHomography(HomogMat, f0, f1, f0ok, f1ok);
		if(f0ok && f1ok) {
			double temp = sqrtf(f0 * f1);
			cout << "fff=" << temp << endl;
			all_focals.push_back(sqrtf(f0 * f1));
		}
	}

	if(all_focals.size() >= num_images - 1) {
		std::sort(all_focals.begin(), all_focals.end());
		if(all_focals.size() % 2 == 1) {
			median = all_focals[all_focals.size() / 2];
		} else {
			median = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5f;
		}
	} 
	
	else {
		throw out_of_range("123");
		double focals_sum = 0;
		focals_sum += img1Size + img2Size;
		median = focals_sum / num_images;
	}
	//cout << "ft = " << ft << endl;
	return median;
}
// 估算焦距
void estimateFocal(const vector<double> &HomogMat, double& focals) {
	if (!HomogMat.empty()) {
		double f0, f1;
		bool f0ok, f1ok;
		focalsFromHomography(HomogMat, f0, f1, f0ok, f1ok);
		if (f0ok && f1ok) focals = std::sqrt(f0 * f1);
	}
}



// 對齊取得第二張圖偏移量
void getWarpOffset(const ImgRaw &imgA, const ImgRaw &imgB,
	Feature const* const* good_match, int gm_num,
	int &dx, int &dy, float FL)
{
	Timer t, t1;

	// 中間值.
	const float&& mid_x1 = (float)imgA.width / 2.f;
	const float&& mid_x2 = (float)imgB.width / 2.f;
	const float&& mid_y1 = (float)imgA.height / 2.f;
	const float&& mid_y2 = (float)imgB.height / 2.f;
	// 先算平方.
	const float& fL1 = FL;
	const float& fL2 = FL;
	const float&& fL1_pow = pow(fL1, 2);
	const float&& fL2_pow = pow(fL2, 2);

	
	int cal_dx(0), cal_dy(0);
//#pragma omp parallel for reduction(+:cal_dx) reduction(+:cal_dy)
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
		int x2 = (int)(fL2*theta2 + mid_x2 + imgA.width +.5);
		int y2 = (int)(fL2*h2 + mid_y2 +.5);
		// 累加座標.
		int distX = x2 - x1;
		int distY = imgA.height - y1 + y2;
		cal_dx += distX;
		cal_dy += distY;
	}

	// 平均座標.
	int avg_dx = (float)cal_dx / (float)(gm_num-1);
	int avg_dy = (float)cal_dy / (float)(gm_num-1);

	// 修正座標(猜測是4捨5入哪裡怎樣沒寫好才變成這樣).
	if(avg_dx % 2 == 0){
		if( imgA.width-avg_dx + 1 <= imgA.width && imgA.width- avg_dx + 1 <= imgB.width){
			//avg_dx += 1;
		} else{
			//avg_dx -= 1;
		}
	} 
	else if(avg_dx % 2 == 1) {
		avg_dx += +0; // 越多右圖越往 <-
	}

	static int num=-1;
	++num;
	if(avg_dy % 2 == 0){
		if(imgA.height-avg_dy + 1 <= imgA.height && imgA.height-avg_dy + 1 <= imgB.height
			and abs((int)imgA.height-avg_dy)>1){
			avg_dy += 1;
			cout << abs(avg_dy) <<"   ############# this Y is up" << num << endl;

		} else{
			avg_dy -= 1;
			cout << abs(avg_dy) <<"	  ########### this Y is dw"<<num << endl;

		}
	} 
	else if(avg_dy % 2 == 1){
		//avg_dy+=0; // 越多右圖越往上
		//avg_dx-=1; // 越多右圖越往上
		cout << "		############ this Y is else"<<num << endl;
	}
	cout << endl;
	// 假如 y 的偏移量大於圖片高
	int xM, yM;
	if(avg_dy > imgA.height) {
		int dyy = -((int)imgA.height - abs((int)imgA.height - avg_dy));
		int xMove = avg_dx;
		int yMove = dyy;
		xM = imgA.width - xMove;
		yM = -(int)imgA.height - yMove;
	} else { // 通常情況
		int xMove = avg_dx;
		int yMove = avg_dy;

		xM = imgA.width - xMove;
		yM = imgA.height - yMove;
	}

	// 輸出座標
	dx=xM;
	dy=yM;
}




