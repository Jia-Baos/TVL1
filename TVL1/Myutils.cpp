#include "Myutils.h"

void movepixels_2d2(cv::Mat src, cv::Mat& dst, cv::Mat Tx, cv::Mat Ty, int interpolation)
{
	//像素重采样实现

	cv::Mat Tx_map(src.size(), CV_32FC1, 0.0);
	cv::Mat Ty_map(src.size(), CV_32FC1, 0.0);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Tx_map = Tx_map.ptr<float>(i);
		float* p_Ty_map = Ty_map.ptr<float>(i);
		for (int j = 0; j < src.cols; j++)
		{
			p_Tx_map[j] = j + Tx.ptr<float>(i)[j];
			p_Ty_map[j] = i + Ty.ptr<float>(i)[j];
		}
	}

	cv::remap(src, dst, Tx_map, Ty_map, interpolation);
}