#pragma once
#ifndef FLOWSHOW_H
#define FLOWSHOW_H

#include <opencv2/opencv.hpp>

#define UNKNOWN_THRESH  1e5
#define EPS 1e-10
#define pi 3.1415926

void flo2img(cv::Mat& flo, cv::Mat& img)
{

	unsigned char ncols;

	unsigned short RY = 15;
	unsigned short YG = 6;
	unsigned short GC = 4;
	unsigned short CB = 11;
	unsigned short BM = 13;
	unsigned short MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	float colorwheel[55][3];
	unsigned short nchans = 3;
	unsigned short col = 0;

	//RY
	for (int i = 0; i < RY; i++)
	{
		colorwheel[col + i][0] = 255;
		colorwheel[col + i][1] = 255 * i / RY;
		colorwheel[col + i][2] = 0;
		//std::cout << colorwheel[i][1] << '\n';
	}
	col += RY;
	//YG
	for (int i = 0; i < YG; i++)
	{
		colorwheel[col + i][0] = 255 - 255 * i / YG;
		colorwheel[col + i][1] = 255;
		colorwheel[col + i][2] = 0;
	}
	col += YG;
	//GC
	for (int i = 0; i < GC; i++)
	{
		colorwheel[col + i][1] = 255;
		colorwheel[col + i][2] = 255 * i / GC;
		colorwheel[col + i][0] = 0;
	}
	col += GC;
	//CB
	for (int i = 0; i < CB; i++)
	{
		colorwheel[col + i][1] = 255 - 255 * i / CB;
		colorwheel[col + i][2] = 255;
		colorwheel[col + i][0] = 0;
	}
	col += CB;
	//BM
	for (int i = 0; i < BM; i++)
	{
		colorwheel[col + i][2] = 255;
		colorwheel[col + i][0] = 255 * i / BM;
		colorwheel[col + i][1] = 0;
	}
	col += BM;
	//MR
	for (int i = 0; i < MR; i++)
	{
		colorwheel[col + i][2] = 255 - 255 * i / MR;
		colorwheel[col + i][0] = 255;
		colorwheel[col + i][1] = 0;
	}

	//std::cout << '\n';
	//for (int i = 0; i < 90; i++)
	//{
	//	for (int j = 0; j < 3; j++)
	//	{
	//		std::cout << colorwheel[i][j] << " | ";
	//	}
	//	std::cout << '\n';
	//}


	int row = flo.rows;
	int cols = flo.cols;
	float max_norm = 1e-10;
	//compute the max norm
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			float* data = flo.ptr<float>(i, j);
			float u = data[0];
			float v = data[1];
			float norm = sqrt(u * u + v * v);
			if (norm > UNKNOWN_THRESH)
			{
				data[0] = 0;
				data[1] = 0;
			}
			else if (norm > max_norm)
			{
				max_norm = norm;
			}
		}
	}
	//calculate the rgb value
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			float* data = flo.ptr<float>(i, j);
			unsigned char* img_data = img.ptr<unsigned char>(i, j);
			float u = data[0];
			float v = data[1];
			float norm = sqrt(u * u + v * v) / max_norm;
			float angle = atan2(-v, -u) / pi;
			float fk = (angle + 1) / 2 * (float(ncols) - 1);
			int k0 = (int)floor(fk);
			int k1 = k0 + 1;
			if (k1 == ncols) {
				k1 = 0;
			}
			float f = fk - k0;
			for (int k = 0; k < 3; k++) {
				float col0 = (colorwheel[k0][k] / 255);
				float col1 = (colorwheel[k1][k] / 255);
				float col3 = (1 - f) * col0 + f * col1;
				if (norm <= 1) {
					col3 = 1 - norm * (1 - col3);
				}
				else {
					col3 *= 0.75;
				}
				img_data[k] = (unsigned char)(255 * col3);
			}

		}
	}
}

void test()
{
	cv::Mat opticalflow = cv::Mat::zeros(cv::Size(100, 100), CV_32FC2);
	int nr = opticalflow.rows; // number of rows
	int nc = opticalflow.cols * opticalflow.channels(); // total number of elements per line
	for (int j = 0; j < nr; j++) {
		float* data = opticalflow.ptr<float>(j);
		for (int i = 0; i < nc; i += 2) {
			data[i] = i / 2 - 50;
			data[i + 1] = j - 50;
		}
	}
	cv::Mat florgb(opticalflow.size(), CV_8UC3);
	flo2img(opticalflow, florgb);
	cv::imshow("rgb", florgb);
	cv::waitKey(0);
}

#endif // !FLOWSHOW_H
