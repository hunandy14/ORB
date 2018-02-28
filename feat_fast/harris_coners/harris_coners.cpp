#include <iostream>
#include <vector>
#include <string>

#include "ImgRaw\Imgraw.hpp"
extern "C" {
#include "fastlib\fast.h"
}

#include "harris_coners.hpp"
using namespace std;



#define HarrisR 2
void harris_coners(const ImgRaw& img, xy** feat, int* size){
	const int r = HarrisR;
	xy* feat_t = new xy[img.width*img.height]{};
	int idx_out = 0;


	int numcorners = *size;
	const xy* feat_123 = *feat;


	vector<float> r_data(img.size());
	for(int k = 0; k < numcorners; k++){
		const int& i = feat_123[k].x;
		const int& j = feat_123[k].y;

		int A = 0, B = 0, C = 0;
		for(int rj = -r + 1; rj < r - 1; rj++){
			for(int ri = -r + 1; ri < r - 1; ri++){
				float dx = img.at2d(j + rj, i + ri + 1) - img.at2d(j + rj, i + ri - 1);
				float dy = img.at2d(j + rj + 1, i + ri) - img.at2d(j + rj - 1, i + ri);
				A += pow(dx, 2);
				B += pow(dy, 2);
				C += dy * dx;
			}
		}
		int detM = A * C + B * B;
		int traceM = A + C;

		const float alpha = 0.04, t = 0.01;
		const float data = detM - alpha * traceM*traceM;
		if(data > t){
			feat_t[idx_out].x = i;
			feat_t[idx_out].y = j;
			if(i==0){
				throw out_of_range("¥X²{0");
			}
			r_data[j*img.width+i] = data;
			++idx_out;
		}
	}

	r_data.resize(idx_out);
	xy* temp = new xy[idx_out];
	copy_n(feat_t, idx_out, temp);
	delete[] feat_t;

	*size = idx_out;
	*feat = temp;
	temp = nullptr;





	// ¹LÂo©P³ò.
	/*
	for(int j = 1, c = 0; j < img.height - 1; j++){
		for(int i = 1; i < img.width - 1; i++, c++){

			int idxp = (j - 1)*img.width + i;
			int idx = j * img.width + i;
			int idxn = (j + 1)*img.width + i;

			if(r_data[idx] != 0){
				if(
					r_data[idx] > r_data[idx + 1] and
					r_data[idx] > r_data[idx - 1] and

					r_data[idx] > r_data[idxp - 1] and
					r_data[idx] > r_data[idxp + 0] and
					r_data[idx] > r_data[idxp + 1] and

					r_data[idx] > r_data[idxn - 1] and
					r_data[idx] > r_data[idxn + 0] and
					r_data[idx] > r_data[idxn + 1]
					){
					h[idx] = 1;
					//feat_harris[idx].x = i;
					//feat_harris[idx].y = j;
					//cout << "*-*-*-*-*" << endl;
					cout << "x=" << i << ", y=" << j << endl;
				} else{
					h[idx] = 0;
				}
			}
		}
		if(c >= idx_out){
			break;
		}
	}*/
}

