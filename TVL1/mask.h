#pragma once
#ifndef MASK_H
#define MASK_H

#include <opencv2/opencv.hpp>

#define BOUNDARY_CONDITION_DIRICHLET 0
#define BOUNDARY_CONDITION_REFLECTING 1
#define BOUNDARY_CONDITION_PERIODIC 2

#define DEFAULT_GAUSSIAN_WINDOW_SIZE 5
#define DEFAULT_BOUNDARY_CONDITION BOUNDARY_CONDITION_REFLECTING

void divergence(
	const cv::Mat v1,	// x component of the vector field
	const cv::Mat v2,	// y component of the vector field
	cv::Mat& div,		// output divergence
	const int nx,		// image width
	const int ny		// image height
);

void forward_gradient(
	const cv::Mat f,	//input image
	cv::Mat& fx,		//computed x derivative
	cv::Mat& fy,		//computed y derivative
	const int nx,		//image width
	const int ny		//image height
);

void centered_gradient(
	const cv::Mat input,//input image
	cv::Mat& dx,		//computed x derivative
	cv::Mat& dy,		//computed y derivative
	const int nx,		//image width
	const int ny		//image height
);

void gaussian(
	cv::Mat& I,			// input/output image
	const int xdim,		// image width
	const int ydim,		// image height
	const double sigma	// Gaussian sigma
);

#endif // !MASK_H
