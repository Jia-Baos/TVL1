#ifndef TVL1FLOW_LIB_C
#define TVL1FLOW_LIB_C

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mask.h"
#include "zoom.h"
#include "bicubic_interpolation.h"

#define MAX_ITERATIONS 300
#define PRESMOOTHING_SIGMA 0.8
#define GRAD_IS_ZERO 1E-10

/**
 * Implementation of the Zach, Pock and Bischof dual TV-L1 optic flow method
 *
 * see reference:
 *  [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime
 *      TV-L1 Optical Flow", In Proceedings of Pattern Recognition (DAGM),
 *      Heidelberg, Germany, pp. 214-223, 2007
 *
 *
 * Details on the total variation minimization scheme can be found in:
 *  [2] A. Chambolle, "An Algorithm for Total Variation Minimization and
 *      Applications", Journal of Mathematical Imaging and Vision, 20: 89-97, 2004
 **/


 /**
  *
  * Function to compute the optical flow in one scale
  *
  **/
void Dual_TVL1_optic_flow(
	const cv::Mat I0,           // source image
	const cv::Mat I1,           // target image
	cv::Mat& u1,           // x component of the optical flow
	cv::Mat& u2,           // y component of the optical flow
	const int   nx,      // image width
	const int   ny,      // image height
	const float tau,     // time step
	const float lambda,  // weight parameter for the data term
	const float theta,   // weight parameter for (u - v)²
	const int   warps,   // number of warpings per scale
	const float epsilon, // tolerance for numerical convergence
	const bool  verbose  // enable/disable the verbose mode
)
{
	const float l_t = lambda * theta;

	// initailize the matriex
	cv::Mat I1x = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat I1y = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat I1w = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat I1wx = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat I1wy = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat rho_c = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat v1 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat v2 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat p11 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat p12 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat p21 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat p22 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat div = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat grad = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat div_p1 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat div_p2 = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat u1x = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat u1y = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat u2x = cv::Mat::zeros(nx, ny, CV_32FC1);
	cv::Mat u2y = cv::Mat::zeros(nx, ny, CV_32FC1);

	float* I0Data = (float*)I0.data;
	float* u1Data = (float*)u1.data;
	float* u2Data = (float*)u2.data;

	float* I1xData = (float*)I1x.data;
	float* I1yData = (float*)I1y.data;
	float* I1wData = (float*)I1w.data;
	float* I1wxData = (float*)I1wx.data;
	float* I1wyData = (float*)I1wy.data;
	float* rho_cData = (float*)rho_c.data;
	float* v1Data = (float*)v1.data;
	float* v2Data = (float*)v2.data;
	float* p11Data = (float*)p11.data;
	float* p12Data = (float*)p12.data;
	float* p21Data = (float*)p21.data;
	float* p22Data = (float*)p22.data;
	float* divData = (float*)div.data;
	float* gradData = (float*)grad.data;
	float* div_p1Data = (float*)div_p1.data;
	float* div_p2Data = (float*)div_p2.data;
	float* u1xData = (float*)u1x.data;
	float* u1yData = (float*)u1y.data;
	float* u2xData = (float*)u2x.data;
	float* u2yData = (float*)u2y.data;

	centered_gradient(I1, I1x, I1y, nx, ny);

	for (int warpings = 0; warpings < warps; warpings++)
	{
		// compute the warping of the target image and its derivatives
		bicubic_interpolation_warp(I1, u1, u2, I1w, nx, ny, true);
		bicubic_interpolation_warp(I1x, u1, u2, I1wx, nx, ny, true);
		bicubic_interpolation_warp(I1y, u1, u2, I1wy, nx, ny, true);

		for (int i = 0; i < ny; i++)
			for (int j = 0; j < nx; j++)
			{
				const float Ix2 = *(I1wxData + i * nx + j) * *(I1wxData + i * nx + j);
				const float Iy2 = *(I1wyData + i * nx + j) * *(I1wyData + i * nx + j);

				// store the |Grad(I1)|^2
				*(gradData + i * nx + j) = (Ix2 + Iy2);

				// compute the constant part of the rho function
				*(rho_cData + i * nx + j) = (*(I1wData + i * nx + j) - *(I1wxData + i * nx + j) * *(u1Data + i * nx + j)
					- *(I1wyData + i * nx + j) * *(u2Data + i * nx + j) - *(I0Data + +i * nx + j));
			}

		int n = 0;
		float error = FLT_MAX;
		while (error > epsilon * epsilon && n < MAX_ITERATIONS)
		{
			n++;
			// estimate the values of the variable (v1, v2)
			// (thresholding opterator TH)
			for (int i = 0; i < ny; i++)
				for (int j = 0; j < nx; j++)
				{
					const float rho = *(rho_cData + i * nx + j)
						+ (*(I1wxData + i * nx + j) * *(u1Data + i * nx + j) + *(I1wyData + i * nx + j) * *(u2Data + i * nx + j));

					float d1, d2;

					if (rho < -l_t * *(gradData + i * nx + j))
					{
						d1 = l_t * *(I1wxData + i * nx + j);
						d2 = l_t * *(I1wyData + i * nx + j);
					}
					else
					{
						if (rho > l_t * *(gradData + i * nx + j))
						{
							d1 = -l_t * *(I1wxData + i * nx + j);
							d2 = -l_t * *(I1wyData + i * nx + j);
						}
						else
						{
							if (*(gradData + i * nx + j) < GRAD_IS_ZERO)
								d1 = d2 = 0;
							else
							{
								float fi = -rho / *(gradData + i * nx + j);
								d1 = fi * *(I1wxData + i * nx + j);
								d2 = fi * *(I1wyData + i * nx + j);
							}
						}
					}

					*(v1Data + i * nx + j) = *(u1Data + i * nx + j) + d1;
					*(v2Data + i * nx + j) = *(u2Data + i * nx + j) + d2;
				}

			// compute the divergence of the dual variable (p1, p2)
			divergence(p11, p12, div_p1, nx, ny);
			divergence(p21, p22, div_p2, nx, ny);

			// estimate the values of the optical flow (u1, u2)
			error = 0.0;

			for (int i = 0; i < ny; i++)
				for (int j = 0; j < nx; j++)
				{
					const float u1k = *(u1Data + i * nx + j);
					const float u2k = *(u2Data + i * nx + j);

					*(u1Data + i * nx + j) = *(v1Data + i * nx + j) + theta * *(div_p1Data + i * nx + j);
					*(u2Data + i * nx + j) = *(v2Data + i * nx + j) + theta * *(div_p2Data + i * nx + j);

					error += (*(u1Data + i * nx + j) - u1k) * (*(u1Data + i * nx + j) - u1k) +
						(*(u2Data + i * nx + j) - u2k) * (*(u2Data + i * nx + j) - u2k);
				}
			error /= nx * ny;

			// compute the gradient of the optical flow (Du1, Du2)
			forward_gradient(u1, u1x, u1y, nx, ny);
			forward_gradient(u2, u2x, u2y, nx, ny);

			// estimate the values of the dual variable (p1, p2)

			for (int i = 0; i < ny; i++)
				for (int j = 0; j < nx; j++)
				{
					const float taut = tau / theta;
					const float g1 = hypot(*(u1xData + i * nx + j), *(u1yData + i * nx + j));
					const float g2 = hypot(*(u2xData + i * nx + j), *(u2yData + i * nx + j));
					const float ng1 = 1.0 + taut * g1;
					const float ng2 = 1.0 + taut * g2;

					*(p11Data + i * nx + j) = (*(p11Data + i * nx + j) + taut * *(u1xData + i * nx + j)) / ng1;
					*(p12Data + i * nx + j) = (*(p12Data + i * nx + j) + taut * *(u1yData + i * nx + j)) / ng1;
					*(p21Data + i * nx + j) = (*(p21Data + i * nx + j) + taut * *(u2xData + i * nx + j)) / ng2;
					*(p22Data + i * nx + j) = (*(p22Data + i * nx + j) + taut * *(u2yData + i * nx + j)) / ng2;
				}
		}

		if (verbose)
			std::cout << stderr
			<< "Warping: " << warpings
			<< "; Iterations: " << n
			<< "; Error: " << error << std::endl;
	}
}

/**
 *
 * Compute the max and min of an array
 *
 **/
static void getminmax(
	float* min,			// output min
	float* max,			// output max
	const cv::Mat x		// input array
)
{
	float* xData = (float*)x.data;
	*min = *max = *(xData);

	for (int i = 0; i < x.rows; i++)
		for (int j = 0; j < x.cols; j++)
		{
			if (*(xData + i * x.cols + j) < *min)
				*min = *(xData + i * x.cols + j);
			if (*(xData + i * x.cols + j) > *max)
				*max = *(xData + i * x.cols + j);
		}
}

/**
 *
 * Function to normalize the images between 0 and 255
 *
 **/
void image_normalization(
	const cv::Mat I0,	// input image0
	const cv::Mat I1,	// input image1
	cv::Mat& I0n,       // normalized output image0
	cv::Mat& I1n		// normalized output image1
)
{
	float max0, max1, min0, min1;

	// initalize the images and pointer of images
	I0n = cv::Mat::zeros(I0.size(), CV_32FC1);
	I1n = cv::Mat::zeros(I1.size(), CV_32FC1);
	float* I0Data = (float*)I0.data;
	float* I1Data = (float*)I1.data;
	float* I0nData = (float*)I0n.data;
	float* I1nData = (float*)I1n.data;

	// obtain the max and min of each image
	getminmax(&min0, &max0, I0);
	getminmax(&min1, &max1, I1);

	// obtain the max and min of both images
	const float max = (max0 > max1) ? max0 : max1;
	const float min = (min0 < min1) ? min0 : min1;
	const float den = max - min;

	if (den > 0)
		// normalize both images
		for (int i = 0; i < I0.rows; i++)
			for (int j = 0; j < I0.cols; j++) {
				*(I0nData + i * I0n.cols + j) = 255.0 * (*(I0Data + i * I0.cols + j) - min) / den;
				*(I1nData + i * I1n.cols + j) = 255.0 * (*(I1Data + i * I1.cols + j) - min) / den;
			}
	else
		// copy the original images
		for (int i = 0; i < I0.rows; i++)
			for (int j = 0; j < I0.cols; j++) {
				*(I0nData + i * I0n.cols + j) = *(I0Data + i * I0.cols + j);
				*(I1nData + i * I1n.cols + j) = *(I1Data + i * I1.cols + j);
			}
}


/**
 *
 * Function to compute the optical flow using multiple scales
 *
 **/
void Dual_TVL1_optic_flow_multiscale(
	const cv::Mat I0,	// source image
	const cv::Mat I1,	// target image
	cv::Mat& u1,		// x component of the optical flow
	cv::Mat& u2,		// y component of the optical flow
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
)
{
	// allocate memory for the pyramid structure
	//initalize the width of images, images and optical flow in pyramid
	std::vector<int> nx(nscales), ny(nscales);
	std::vector<cv::Mat> I0s(nscales), I1s(nscales);
	std::vector<cv::Mat> u1s(nscales), u2s(nscales);

	u1s[0] = u1, u2s[0] = u2;
	nx[0] = nxx, ny[0] = nyy;

	// normalize the images between 0 and 255
	//I0s[0] = cv::Mat::zeros(nxx, nyy, CV_32FC1);
	//I1s[0] = cv::Mat::zeros(nxx, nyy, CV_32FC1);
	image_normalization(I0, I1, I0s[0], I1s[0]);

	// pre-smooth the original images
	gaussian(I0s[0], nx[0], ny[0], PRESMOOTHING_SIGMA);
	gaussian(I1s[0], nx[0], ny[0], PRESMOOTHING_SIGMA);

	// create the scales
	for (int s = 1; s < nscales; s++)
	{
		zoom_size(nx[s - 1], ny[s - 1], &nx[s], &ny[s], zfactor);

		// zoom in the images to create the pyramidal structure
		//I0s[s] = cv::Mat::zeros(nx[s], ny[s], CV_32FC1);
		//I1s[s] = cv::Mat::zeros(nx[s], ny[s], CV_32FC1);
		zoom_out(I0s[s - 1], I0s[s], nx[s - 1], ny[s - 1], zfactor);
		zoom_out(I1s[s - 1], I1s[s], nx[s - 1], ny[s - 1], zfactor);
	}

	// initialize the flow at the coarsest scale
	u1s[nscales - 1] = cv::Mat::zeros(nx[nscales - 1], ny[nscales - 1], CV_32FC1);
	u2s[nscales - 1] = cv::Mat::zeros(nx[nscales - 1], ny[nscales - 1], CV_32FC1);

	// pyramidal structure for computing the optical flow
	for (int s = nscales - 1; s >= 0; s--)
	{
		if (verbose)
			std::cout << stderr
			<< "Scale: " << s
			<< "; Width: " << nx[s]
			<< "; Height: " << ny[s] << std::endl;

		// compute the optical flow at the current scale
		Dual_TVL1_optic_flow(
			I0s[s], I1s[s], u1s[s], u2s[s], nx[s], ny[s],
			tau, lambda, theta, warps, epsilon, verbose
		);

		// if this was the last scale, finish now
		if (!s) break;

		// otherwise, upsample the optical flow

		// zoom the optical flow for the next finer scale
		zoom_in(u1s[s], u1s[s - 1], nx[s], ny[s], nx[s - 1], ny[s - 1]);
		zoom_in(u2s[s], u2s[s - 1], nx[s], ny[s], nx[s - 1], ny[s - 1]);

		// scale the optical flow with the appropriate zoom factor
		float* u1sData = (float*)u1s[s - 1].data;
		float* u2sData = (float*)u2s[s - 1].data;
		for (int i = 0; i < ny[s - 1]; i++)
			for (int j = 0; j < nx[s - 1]; j++)
			{
				*(u1sData + i * nx[s - 1] + j) *= (float)1.0 / zfactor;
				*(u2sData + i * nx[s - 1] + j) *= (float)1.0 / zfactor;
			}
	}
}

#endif // !TVL1FLOW_LIB_C
