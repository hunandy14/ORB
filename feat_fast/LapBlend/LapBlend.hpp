/*****************************************************************
Name :
Date : 2018/04/12
By   : CharlotteHonG
Final: 2018/04/12
*****************************************************************/
#pragma once

// 資料結構
struct basic_ImgData {
	std::vector<unsigned char> raw_img;
	uint32_t width;
	uint32_t height;
	uint16_t bits;
};
// ImgData 操作函式
void ImgData_resize(basic_ImgData & dst, int newW, int newH, int bits);
void ImgData_resize(const basic_ImgData & src, basic_ImgData & dst);
void ImgData_write(const basic_ImgData & src, string name);
void ImgData_read(basic_ImgData & dst, std::string name);

// 混合原始圖
void LapBlender(basic_ImgData & dst, const basic_ImgData & src1, const basic_ImgData & src2, double ft, int mx, int my);

// 範例測試
void LapBlend_Tester();
