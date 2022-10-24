﻿#pragma once
#ifndef TVL1FLOW_LIB_H
#define TVL1FLOW_LIB_H

#include <opencv2/opencv.hpp>

#define MAX_ITERATIONS 300
#define PRESMOOTHING_SIGMA 0.8
#define GRAD_IS_ZERO 1E-10

void Dual_TVL1_optic_flow_multiscale(
	const cv::Mat I0,           // source image
	const cv::Mat I1,           // target image
	cv::Mat& u1,           // x component of the optical flow
	cv::Mat& u2,           // y component of the optical flow
	const int   nxx,     // image width
	const int   nyy,     // image height
	const float tau,     // time step
	const float lambda,  // weight parameter for the data term
	const float theta,   // weight parameter for (u - v)²
	const int   nscales, // number of scales
	const float zfactor, // factor for building the image piramid
	const int   warps,   // number of warpings per scale
	const float epsilon, // tolerance for numerical convergence
	const bool  verbose  // enable/disable the verbose mode
);

#endif // !TVL1FLOW_LIB_H

