/*****************************************************************
Name : 
Date : 2018/03/01
By   : CharlotteHonG
Final: 2018/03/19
*****************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <opencv2\opencv.hpp> 
using namespace cv;


#include "Imgraw.hpp"
extern "C" {
#include "fastlib\fast.h"
}

#include "feat.hpp"
#include "harris_coners.hpp"
using namespace std;

#define HarrisR 2
void harris_coners(const ImgRaw& img, Feat& feat){
	const int r = HarrisR;
	Feat feat_t(feat.size());

	int idx_out = 0;

	const xy* feat_old = feat;
	vector<float> r_data(img.size());
	for(int k = 0; k < feat.size(); k++){
		const int& i = feat_old[k].x;
		const int& j = feat_old[k].y;

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
				throw out_of_range("出現0");
			}
			r_data[j*img.width+i] = data;
			++idx_out;
		}
	}
	r_data.resize(idx_out);
	// 縮小.
	feat.~Feat();

	// todo 這裡為什麼不能刪除
	xy* temp = new xy[idx_out];
	for(size_t i = 0; i < idx_out; i++) {
		temp[i] = feat_t.feat[i];
	}
	feat.len = idx_out;
	feat.feat = temp;
	temp = nullptr;

	cout << "idx_out==" << idx_out << endl;



	// 過濾周圍.
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


enum { MINEIGENVAL=0, HARRIS=1, EIGENVALSVECS=2 };
//====================================================================================
static void calcMinEigenVal( const Mat& _cov, Mat& _dst )
{
	int i, j;
	Size size = _cov.size();
#if CV_TRY_AVX
	bool haveAvx = CV_CPU_HAS_SUPPORT_AVX;
#endif
#if CV_SIMD128
	bool haveSimd = hasSIMD128();
#endif

	if( _cov.isContinuous() && _dst.isContinuous() )
	{
		size.width *= size.height;
		size.height = 1;
	}

	for( i = 0; i < size.height; i++ )
	{
		const float* cov = _cov.ptr<float>(i);
		float* dst = _dst.ptr<float>(i);
#if CV_TRY_AVX
		if( haveAvx )
			j = calcMinEigenValLine_AVX(cov, dst, size.width);
		else
#endif // CV_TRY_AVX
			j = 0;

#if CV_SIMD128
		if( haveSimd )
		{
			v_float32x4 half = v_setall_f32(0.5f);
			for( ; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes )
			{
				v_float32x4 v_a, v_b, v_c, v_t;
				v_load_deinterleave(cov + j*3, v_a, v_b, v_c);
				v_a *= half;
				v_c *= half;
				v_t = v_a - v_c;
				v_t = v_muladd(v_b, v_b, (v_t * v_t));
				v_store(dst + j, (v_a + v_c) - v_sqrt(v_t));
			}
		}
#endif // CV_SIMD128

		for( ; j < size.width; j++ )
		{
			float a = cov[j*3]*0.5f;
			float b = cov[j*3+1];
			float c = cov[j*3+2]*0.5f;
			dst[j] = (float)((a + c) - std::sqrt((a - c)*(a - c) + b*b));
		}
	}
}
static void calcHarris( const Mat& _cov, Mat& _dst, double k )
{
	int i, j;
	Size size = _cov.size();
#if CV_TRY_AVX
	bool haveAvx = CV_CPU_HAS_SUPPORT_AVX;
#endif
#if CV_SIMD128
	bool haveSimd = hasSIMD128();
#endif

	if( _cov.isContinuous() && _dst.isContinuous() )
	{
		size.width *= size.height;
		size.height = 1;
	}

	for( i = 0; i < size.height; i++ )
	{
		const float* cov = _cov.ptr<float>(i);
		float* dst = _dst.ptr<float>(i);

#if CV_TRY_AVX
		if( haveAvx )
			j = calcHarrisLine_AVX(cov, dst, k, size.width);
		else
#endif // CV_TRY_AVX
			j = 0;

#if CV_SIMD128
		if( haveSimd )
		{
			v_float32x4 v_k = v_setall_f32((float)k);

			for( ; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes )
			{
				v_float32x4 v_a, v_b, v_c;
				v_load_deinterleave(cov + j * 3, v_a, v_b, v_c);

				v_float32x4 v_ac_bb = v_a * v_c - v_b * v_b;
				v_float32x4 v_ac = v_a + v_c;
				v_float32x4 v_dst = v_ac_bb - v_k * v_ac * v_ac;
				v_store(dst + j, v_dst);
			}
		}
#endif // CV_SIMD128

		for( ; j < size.width; j++ )
		{
			float a = cov[j*3];
			float b = cov[j*3+1];
			float c = cov[j*3+2];
			dst[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
		}
	}
}
static void eigen2x2( const float* cov, float* dst, int n )
{
	for( int j = 0; j < n; j++ )
	{
		double a = cov[j*3];
		double b = cov[j*3+1];
		double c = cov[j*3+2];

		double u = (a + c)*0.5;
		double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
		double l1 = u + v;
		double l2 = u - v;

		double x = b;
		double y = l1 - a;
		double e = fabs(x);

		if( e + fabs(y) < 1e-4 )
		{
			y = b;
			x = l1 - c;
			e = fabs(x);
			if( e + fabs(y) < 1e-4 )
			{
				e = 1./(e + fabs(y) + FLT_EPSILON);
				x *= e, y *= e;
			}
		}

		double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
		dst[6*j] = (float)l1;
		dst[6*j + 2] = (float)(x*d);
		dst[6*j + 3] = (float)(y*d);

		x = b;
		y = l2 - a;
		e = fabs(x);

		if( e + fabs(y) < 1e-4 )
		{
			y = b;
			x = l2 - c;
			e = fabs(x);
			if( e + fabs(y) < 1e-4 )
			{
				e = 1./(e + fabs(y) + FLT_EPSILON);
				x *= e, y *= e;
			}
		}

		d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
		dst[6*j + 1] = (float)l2;
		dst[6*j + 4] = (float)(x*d);
		dst[6*j + 5] = (float)(y*d);
	}
}
static void calcEigenValsVecs( const Mat& _cov, Mat& _dst )
{
	Size size = _cov.size();
	if( _cov.isContinuous() && _dst.isContinuous() )
	{
		size.width *= size.height;
		size.height = 1;
	}

	for( int i = 0; i < size.height; i++ )
	{
		const float* cov = _cov.ptr<float>(i);
		float* dst = _dst.ptr<float>(i);

		eigen2x2(cov, dst, size.width);
	}
}
//=================================================================
void Sobel( InputArray _src, OutputArray _dst, int ddepth, 
	int dx, int dy, int ksize = 3,
	double scale = 1, double delta = 0,
	int borderType = BORDER_DEFAULT)
{

	int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
	if (ddepth < 0)
		ddepth = sdepth;
	int dtype = CV_MAKE_TYPE(ddepth, cn);
	_dst.create( _src.size(), dtype );

	int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

	Mat kx, ky;
	getDerivKernels( kx, ky, dx, dy, ksize, false, ktype );
	if( scale != 1 )
	{
		// usually the smoothing part is the slowest to compute,
		// so try to scale it instead of the faster differentiating part
		if( dx == 0 )
			kx *= scale;
		else
			ky *= scale;
	}

	Mat src = _src.getMat();
	Mat dst = _dst.getMat();

	Point ofs;
	Size wsz(src.cols, src.rows);
	if(!(borderType & BORDER_ISOLATED))
		src.locateROI( wsz, ofs );
	sepFilter2D(src, dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}
static void getScharrKernels( OutputArray _kx, OutputArray _ky,
	int dx, int dy, bool normalize, int ktype )
{
	const int ksize = 3;

	CV_Assert( ktype == CV_32F || ktype == CV_64F );
	_kx.create(ksize, 1, ktype, -1, true);
	_ky.create(ksize, 1, ktype, -1, true);
	Mat kx = _kx.getMat();
	Mat ky = _ky.getMat();

	CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );

	for( int k = 0; k < 2; k++ )
	{
		Mat* kernel = k == 0 ? &kx : &ky;
		int order = k == 0 ? dx : dy;
		int kerI[3];

		if( order == 0 )
			kerI[0] = 3, kerI[1] = 10, kerI[2] = 3;
		else if( order == 1 )
			kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;

		Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
		double scale = !normalize || order == 1 ? 1. : 1./32;
		temp.convertTo(*kernel, ktype, scale);
	}
}
void Scharr( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
	double scale, double delta, int borderType )
{

		int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
	if (ddepth < 0)
		ddepth = sdepth;
	int dtype = CV_MAKETYPE(ddepth, cn);
	_dst.create( _src.size(), dtype );

	int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

	Mat kx, ky;
	getScharrKernels( kx, ky, dx, dy, false, ktype );
	if( scale != 1 )
	{
		// usually the smoothing part is the slowest to compute,
		// so try to scale it instead of the faster differentiating part
		if( dx == 0 )
			kx *= scale;
		else
			ky *= scale;
	}


		Mat src = _src.getMat();
	Mat dst = _dst.getMat();

	Point ofs;
	Size wsz(src.cols, src.rows);
	if(!(borderType & BORDER_ISOLATED))
		src.locateROI( wsz, ofs );

	sepFilter2D( src, dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}
//=================================================================
static void
cornerEigenValsVecs( const Mat& src, Mat& eigenv, int block_size,
	int aperture_size, int op_type, double k=0.,
	int borderType=BORDER_DEFAULT )
{
	int depth = src.depth();
	double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
	if( aperture_size < 0 )
		scale *= 2.0;
	if( depth == CV_8U )
		scale *= 255.0;
	scale = 1.0/scale;

	CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );

	Mat Dx, Dy;
	if( aperture_size > 0 )
	{
		::Sobel( src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType );
		::Sobel( src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType );
	}
	else
	{
		::Scharr( src, Dx, CV_32F, 1, 0, scale, 0, borderType );
		::Scharr( src, Dy, CV_32F, 0, 1, scale, 0, borderType );
	}

	Size size = src.size();
	Mat cov( size, CV_32FC3 );
	int i, j;

	for( i = 0; i < size.height; i++ )
	{
		float* cov_data = cov.ptr<float>(i);
		const float* dxdata = Dx.ptr<float>(i);
		const float* dydata = Dy.ptr<float>(i);

			j = 0;

		for( ; j < size.width; j++ )
		{
			float dx = dxdata[j];
			float dy = dydata[j];

			cov_data[j*3] = dx*dx;
			cov_data[j*3+1] = dx*dy;
			cov_data[j*3+2] = dy*dy;
		}
	}

	boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
		Point(-1,-1), false, borderType );

	if( op_type == MINEIGENVAL )
		::calcMinEigenVal( cov, eigenv );
	else if( op_type == HARRIS )
		::calcHarris( cov, eigenv, k );
	else if( op_type == EIGENVALSVECS )
		::calcEigenValsVecs( cov, eigenv );
}
static void cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType = BORDER_DEFAULT )
{
	Mat src = _src.getMat();
	_dst.create( src.size(), CV_32FC1 );
	Mat dst = _dst.getMat();
	cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
}
//====================================================================================
void goodFeaturesToTrack2( InputArray _image, OutputArray _corners,
	int maxCorners, double qualityLevel, double minDistance,
	InputArray _mask, int blockSize, int gradientSize,
	bool useHarrisDetector, double harrisK )
{
	//如果需要对_image全图操作，则给_mask传入cv::Mat()，否则传入感兴趣区域  
	Mat image = _image.getMat(), mask = _mask.getMat();

	CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);  //对参数有一些基本要求  
	CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

	Mat eig, tmp;   //eig存储每个像素协方差矩阵的最小特征值，tmp用来保存经膨胀后的eig  
	if (useHarrisDetector)
		::cornerHarris(image, eig, blockSize, 3, harrisK); //blockSize是计算2*2协方差矩阵的窗口大小，sobel算子窗口为3，harrisK是计算Harris角点时需要的值  
	else
		cornerMinEigenVal(image, eig, blockSize, 3);  //计算每个像素对应的协方差矩阵的最小特征值，保存在eig中  

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal, 0, 0, mask);   //maxVal保存了eig的最大值  
	threshold(eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO);  //阈值设置为maxVal乘以qualityLevel，大于此阈值的保持不变，小于此阈值的都设为0  

																 //默认用3*3的核膨胀，膨胀之后，除了局部最大值点和原来相同，其它非局部最大值点被    
																 //3*3邻域内的最大值点取代，如不理解，可看一下灰度图像的膨胀原理    
	dilate(eig, tmp, Mat());  //tmp中保存了膨胀之后的eig  

	Size imgsize = image.size();

	vector<const float*> tmpCorners;  //存放粗选出的角点地址  

									  // collect list of pointers to features - put them into temporary image   
	for (int y = 1; y < imgsize.height - 1; y++) {
		const float* eig_data = (const float*)eig.ptr(y);  //获得eig第y行的首地址  
		const float* tmp_data = (const float*)tmp.ptr(y);  //获得tmp第y行的首地址  
		const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

		for (int x = 1; x < imgsize.width - 1; x++) {
			float val = eig_data[x];
			if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))  //val == tmp_data[x]说明这是局部极大值  
				tmpCorners.push_back(eig_data + x);  //保存其位置  
		}
	}

	//-----------此分割线以上是根据特征值粗选出的角点，我们称之为弱角点----------//  
	//-----------此分割线以下还要根据minDistance进一步筛选角点，仍然能存活下来的我们称之为强角点----------//  

	struct greaterThanPtr {
		bool operator () (const float * a, const float * b) const
			// Ensure a fully deterministic result of the sort
		{
			return (*a > *b) ? true : (*a < *b) ? false : (a > b);
		}
	};
	std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());
	//sort( tmpCorners, greaterThanPtr<float>() );  //按特征值降序排列，注意这一步很重要，后面的很多编程思路都是建立在这个降序排列的基础上  
	vector<Point2f> corners;
	size_t i, j, total = tmpCorners.size(), ncorners = 0;

	//下面的程序有点稍微难理解，需要自己仔细想想  
	if (minDistance >= 1) {
		// Partition the image into larger grids  
		int w = image.cols;
		int h = image.rows;

		const int cell_size = cvRound(minDistance);   //向最近的整数取整  

													  //这里根据cell_size构建了一个矩形窗口grid(虽然下面的grid定义的是vector<vector>，而并不是我们这里说的矩形窗口，但为了便于理解,还是将grid想象成一个grid_width * grid_height的矩形窗口比较好)，除以cell_size说明grid窗口里相差一个像素相当于_image里相差minDistance个像素，至于为什么加上cell_size - 1后面会讲  
		const int grid_width = (w + cell_size - 1) / cell_size;
		const int grid_height = (h + cell_size - 1) / cell_size;

		std::vector<std::vector<Point2f> > grid(grid_width*grid_height);  //vector里面是vector，grid用来保存获得的强角点坐标  

		minDistance *= minDistance;  //平方，方面后面计算，省的开根号  

		for (i = 0; i < total; i++)     // 刚刚粗选的弱角点，都要到这里来接收新一轮的考验  
		{
			int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);  //tmpCorners中保存了角点的地址，eig.data返回eig内存块的首地址  
			int y = (int)(ofs / eig.step);   //角点在原图像中的行  
			int x = (int)((ofs - y*eig.step)/sizeof(float));  //在原图像中的列  

			bool good = true;  //先认为当前角点能接收考验，即能被保留下来  

			int x_cell = x / cell_size;  //x_cell，y_cell是角点（y,x）在grid中的对应坐标  
			int y_cell = y / cell_size;

			int x1 = x_cell - 1;  // (y_cell，x_cell）的4邻域像素  
			int y1 = y_cell - 1;  //现在知道为什么前面grid_width定义时要加上cell_size - 1了吧，这是为了使得（y,x）在grid中的4邻域像素都存在，也就是说(y_cell，x_cell）不会成为边界像素  
			int x2 = x_cell + 1;
			int y2 = y_cell + 1;

			// boundary check，再次确认x1,y1,x2或y2不会超出grid边界  
			x1 = std::max(0, x1);  //比较0和x1的大小  
			y1 = std::max(0, y1);
			x2 = std::min(grid_width-1, x2);
			y2 = std::min(grid_height-1, y2);

			//记住grid中相差一个像素，相当于_image中相差了minDistance个像素  
			for (int yy = y1; yy <= y2; yy++)  // 行  
			{
				for (int xx = x1; xx <= x2; xx++)  //列  
				{
					vector <Point2f> &m = grid[yy*grid_width + xx];  //引用  

					if (m.size())  //如果(y_cell，x_cell)的4邻域像素，也就是(y,x)的minDistance邻域像素中已有被保留的强角点  
					{
						for (j = 0; j < m.size(); j++)   //当前角点周围的强角点都拉出来跟当前角点比一比  
						{
							float dx = x - m[j].x;
							float dy = y - m[j].y;
							//注意如果(y,x)的minDistance邻域像素中已有被保留的强角点，则说明该强角点是在(y,x)之前就被测试过的，又因为tmpCorners中已按照特征值降序排列（特征值越大说明角点越好），这说明先测试的一定是更好的角点，也就是已保存的强角点一定好于当前角点，所以这里只要比较距离，如果距离满足条件，可以立马扔掉当前测试的角点  
							if (dx*dx + dy*dy < minDistance) {
								good = false;
								goto break_out;
							}
						}
					}
				}   // 列  
			}    //行  

		break_out:

			if (good) {
				// printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",  
				//    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);  
				grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

				corners.push_back(Point2f((float)x, (float)y));
				++ncorners;

				if (maxCorners > 0 && (int)ncorners == maxCorners)  //由于前面已按降序排列，当ncorners超过maxCorners的时候跳出循环直接忽略tmpCorners中剩下的角点，反正剩下的角点越来越弱  
					break;
			}
		}
	} else    //除了像素本身，没有哪个邻域像素能与当前像素满足minDistance < 1,因此直接保存粗选的角点  
	{
		for (i = 0; i < total; i++) {
			int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
			int y = (int)(ofs / eig.step);   //粗选的角点在原图像中的行  
			int x = (int)((ofs - y*eig.step)/sizeof(float));  //在图像中的列  

			corners.push_back(Point2f((float)x, (float)y));
			++ncorners;
			if (maxCorners > 0 && (int)ncorners == maxCorners)
				break;
		}
	}

	Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}