/*****************************************************************
Name :
Date : 2018/04/12
By   : CharlotteHonG
Final: 2018/04/12
*****************************************************************/
#pragma once

#include "Raw2Img.hpp"
// ImgData �ާ@�禡
void ImgData_resize(basic_ImgData & dst, int newW, int newH, int bits);
void ImgData_resize(const basic_ImgData & src, basic_ImgData & dst);
void ImgData_write(const basic_ImgData & src, string name);
void ImgData_read(basic_ImgData & dst, std::string name);

// �V�X��l��
void LapBlender(basic_ImgData & dst, const basic_ImgData & src1, const basic_ImgData & src2, double ft, int mx, int my);

// �d�Ҵ���
void LapBlend_Tester();
