#ifndef ZOOM_C
#define ZOOM_C

#include <cmath>

#include "mask.h"
#include "bicubic_interpolation.h"

#define ZOOM_SIGMA_ZERO 0.6

/**
  *
  * Compute the size of a zoomed image from the zoom factor
  *
**/
void zoom_size(
	int nx,      // width of the orignal image
	int ny,      // height of the orignal image
	int* nxx,    // width of the zoomed image
	int* nyy,    // height of the zoomed image
	float factor // zoom factor between 0 and 1
)
{
	//compute the new size corresponding to factor
	//we add 0.5 for rounding off to the closest number
	*nxx = (int)((float)nx * factor + 0.5);
	*nyy = (int)((float)ny * factor + 0.5);
}

/**
  *
  * Downsample an image
  *
**/
void zoom_out(
	const cv::Mat I,	// input image
	cv::Mat& Iout,		// output image
	const int nx,		// image width
	const int ny,		// image height
	const float factor	// zoom factor between 0 and 1
)
{
	// temporary working image
	cv::Mat Is = I.clone();

	// compute the size of the zoomed image
	int nxx, nyy;
	zoom_size(nx, ny, &nxx, &nyy, factor);

	// compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * sqrt(1.0 / (factor * factor) - 1.0);

	// pre-smooth the image
	gaussian(Is, nx, ny, sigma);

	// re-sample the image using bicubic interpolation
	Iout = cv::Mat::zeros(nxx, nyy, CV_32FC1);
	for (int i1 = 0; i1 < nyy; i1++)
		for (int j1 = 0; j1 < nxx; j1++)
		{
			const float i2 = (float)i1 / factor;
			const float j2 = (float)j1 / factor;

			float g = bicubic_interpolation_at(Is, j2, i2, nx, ny, false);
			
			float* IoutData = (float*)Iout.data;
			*(IoutData + i1 * nxx + j1) = g;
		}
}


/**
  *
  * Function to upsample the image
  *
**/
void zoom_in(
	const cv::Mat I,	// input image
	cv::Mat& Iout,		// output image
	int nx,				// width of the original image
	int ny,				// height of the original image
	int nxx,			// width of the zoomed image
	int nyy				// height of the zoomed image
)
{
	// temporary working image
	cv::Mat Is = I.clone();

	// intialize the temp varible of Iout, otherwise the memory maybe malloc wrongly
	cv::Mat Iout = cv::Mat::zeros(nxx, nyy, CV_32FC1);
	float* IoutData = (float*)Iout.data;

	// compute the zoom factor
	const float factorx = ((float)nxx / nx);
	const float factory = ((float)nyy / ny);

	// re-sample the image using bicubic interpolation
	for (int i1 = 0; i1 < nyy; i1++)
		for (int j1 = 0; j1 < nxx; j1++)
		{
			float i2 = (float)i1 / factory;
			float j2 = (float)j1 / factorx;

			float g = bicubic_interpolation_at(Is, j2, i2, nx, ny, false);
			*(IoutData + i1 * nxx + j1) = g;
		}
}

#endif // !ZOOM_C
