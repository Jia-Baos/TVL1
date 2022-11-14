#pragma once
#ifndef MYUTILS_H
#define MYUTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>

void movepixels_2d2(cv::Mat src, cv::Mat& dst, cv::Mat Tx, cv::Mat Ty, int interpolation);

#endif // !MYUTILS_H
