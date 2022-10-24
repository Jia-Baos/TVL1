#ifndef MASK_C
#define MASK_C

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

#define BOUNDARY_CONDITION_DIRICHLET 0
#define BOUNDARY_CONDITION_REFLECTING 1
#define BOUNDARY_CONDITION_PERIODIC 2

#define DEFAULT_GAUSSIAN_WINDOW_SIZE 5
#define DEFAULT_BOUNDARY_CONDITION BOUNDARY_CONDITION_REFLECTING


/**
 *
 * Details on how to compute the divergence and the grad(u) can be found in:
 * [2] A. Chambolle, "An Algorithm for Total Variation Minimization and
 * Applications", Journal of Mathematical Imaging and Vision, 20: 89-97, 2004
 *
 **/


 /**
  *
  * Function to compute the divergence with backward differences
  * (see [2] for details)
  *
  **/
void divergence(
	const cv::Mat v1,	// x component of the vector field
	const cv::Mat v2,	// y component of the vector field
	cv::Mat& div,		// output divergence
	const int nx,		// image width
	const int ny		// image height
)
{
	// intialize the pointer of images
	div = cv::Mat::zeros(nx, ny, CV_32FC1);
	float* v1Data = (float*)v1.data;
	float* v2Data = (float*)v2.data;
	float* divData = (float*)div.data;
	const int step = v1.step[0] / v1.step[1];

	// compute the divergence on the central body of the image
	for (int i = 1; i < ny - 1; i++)
	{
		for (int j = 1; j < nx - 1; j++)
		{
			const int p = i * nx + j;
			const int p1 = p - 1;
			const int p2 = p - nx;
			const float v1x = *(v1Data + p) - *(v1Data + p1);
			const float v2y = *(v2Data + p) - *(v2Data + p2);

			*(divData + p) = v1x + v2y;
		}
	}

	// compute the divergence on the first and last rows
	for (int j = 1; j < nx - 1; j++)
	{
		const int p = (ny - 1) * nx + j;

		*(divData + j) = *(v1Data + j) - *(v1Data + j - 1) + *(v2Data + j);
		*(divData + p) = *(v1Data + p) - *(v1Data + p - 1) - *(v2Data + p - nx);
	}

	// compute the divergence on the first and last columns
	for (int i = 1; i < ny - 1; i++)
	{
		const int p1 = i * nx;
		const int p2 = (i + 1) * nx - 1;

		*(divData + p1) = *(v1Data + p1) + *(v2Data + p1) - *(v2Data + p1 - nx);
		*(divData + p2) = -*(v1Data + p2 - 1) + *(v2Data + p2) - *(v2Data + p2 - nx);

	}

	*(divData) = *(v1Data)+*(v2Data);
	*(divData + nx - 1) = -*(v1Data + nx - 2) + *(v2Data + nx - 1);
	*(divData + (ny - 1) * nx) = *(v1Data + (ny - 1) * nx) - *(v2Data + (ny - 2) * nx);
	*(divData + ny * nx - 1) = -*(v1Data + ny * nx - 2) - *(v2Data + (ny - 1) * nx - 1);
}


/**
 *
 * Function to compute the gradient with forward differences
 * (see [2] for details)
 *
 **/
void forward_gradient(
	const cv::Mat f,	//input image
	cv::Mat& fx,		//computed x derivative
	cv::Mat& fy,		//computed y derivative
	const int nx,		//image width
	const int ny		//image height
)
{
	// intialize the pointer of images
	fx = cv::Mat::zeros(nx, ny, CV_32FC1);
	fy = cv::Mat::zeros(nx, ny, CV_32FC1);
	float* fData = (float*)f.data;
	float* fxData = (float*)fx.data;
	float* fyData = (float*)fy.data;

	// compute the gradient on the central body of the image
	for (int i = 0; i < ny - 1; i++)
	{
		for (int j = 0; j < nx - 1; j++)
		{
			const int p = i * nx + j;
			const int p1 = p + 1;
			const int p2 = p + nx;

			*(fxData + p) = *(fData + p1) - *(fData + p);
			*(fyData + p) = *(fData + p2) - *(fData + p);
		}
	}

	// compute the gradient on the last row
	for (int j = 0; j < nx - 1; j++)
	{
		const int p = (ny - 1) * nx + j;

		*(fxData + p) = *(fData + p + 1) - *(fData + p);
		*(fyData + p) = 0;
	}

	// compute the gradient on the last column
	for (int i = 1; i < ny; i++)
	{
		const int p = i * nx - 1;

		*(fxData + p) = 0;
		*(fyData + p) = *(fData + p + nx) - *(fData + p);
	}

	*(fxData + ny * nx - 1) = 0;
	*(fyData + ny * nx - 1) = 0;
}


/**
 *
 * Function to compute the gradient with centered differences
 *
 **/
void centered_gradient(
	const cv::Mat input,//input image
	cv::Mat& dx,		//computed x derivative
	cv::Mat& dy,		//computed y derivative
	const int nx,        //image width
	const int ny         //image height
)
{
	// intialize the pointer of images
	dx = cv::Mat::zeros(nx, ny, CV_32FC1);
	dy = cv::Mat::zeros(nx, ny, CV_32FC1);
	float* inputData = (float*)input.data;
	float* dxData = (float*)dx.data;
	float* dyData = (float*)dy.data;

	// compute the gradient on the center body of the image
	for (int i = 1; i < ny - 1; i++)
	{
		for (int j = 1; j < nx - 1; j++)
		{
			const int k = i * nx + j;
			*(dxData + k) = 0.5 * (*(inputData + k + 1) - *(inputData + k - 1));
			*(dyData + k) = 0.5 * (*(inputData + k + nx) - *(inputData + k - nx));
		}
	}

	// compute the gradient on the first and last rows
	for (int j = 1; j < nx - 1; j++)
	{
		*(dxData + j) = 0.5 * (*(inputData + j + 1) - *(inputData + j - 1));
		*(dyData + j) = 0.5 * (*(inputData + j + nx) - *(inputData + j));

		const int k = (ny - 1) * nx + j;

		*(dxData + k) = 0.5 * (*(inputData + k + 1) - *(inputData + k - 1));
		*(dyData + k) = 0.5 * (*(inputData + k) - *(inputData + k - nx));;
	}

	// compute the gradient on the first and last columns
	for (int i = 1; i < ny - 1; i++)
	{
		const int p = i * nx;
		*(dxData + p) = 0.5 * (*(inputData + p + 1) - *(inputData + p));
		*(dyData + p) = 0.5 * (*(inputData + p + nx) - *(inputData + p - nx));

		const int k = (i + 1) * nx - 1;

		*(dxData + k) = 0.5 * (*(inputData + k) - *(inputData + k - 1));
		*(dyData + k) = 0.5 * (*(inputData + k + nx) - *(inputData + k - nx));
	}

	// compute the gradient at the four corners
	*(dxData) = 0.5 * (*(inputData + 1) - *(inputData));
	*(dyData) = 0.5 * (*(inputData + nx) - *(inputData));

	*(dxData + nx - 1) = 0.5 * (*(inputData + nx - 1) - *(inputData + nx - 2));
	*(dyData + nx - 1) = 0.5 * (*(inputData + 2 * nx - 1) - *(inputData + nx - 1));

	*(dxData + (ny - 1) * nx) = 0.5 * (*(inputData + (ny - 1) * nx + 1) - *(inputData + (ny - 1) * nx));
	*(dyData + (ny - 1) * nx) = 0.5 * (*(inputData + (ny - 1) * nx) - *(inputData + (ny - 2) * nx));

	*(dxData + ny * nx - 1) = 0.5 * (*(inputData + ny * nx - 1) - *(inputData + ny * nx - 1 - 1));
	*(dyData + ny * nx - 1) = 0.5 * (*(inputData + ny * nx - 1) - *(inputData + (ny - 1) * nx - 1));
}


/**
 *
 * In-place Gaussian smoothing of an image
 *
 */
void gaussian(
	cv::Mat& I,			// input/output image
	const int xdim,		// image width
	const int ydim,		// image height
	const double sigma	// Gaussian sigma
)
{
	const int boundary_condition = DEFAULT_BOUNDARY_CONDITION;
	const int window_size = DEFAULT_GAUSSIAN_WINDOW_SIZE;

	const double den = 2 * sigma * sigma;
	const int   size = (int)(window_size * sigma) + 1;
	const int   bdx = xdim + size;
	const int   bdy = ydim + size;

	if (boundary_condition && size > xdim) {
		std::cout << stderr << "GaussianSmooth: sigma too large" << std::endl;
		//fprintf(stderr, "GaussianSmooth: sigma too large\n");
		abort();
	}

	// compute the coefficients of the 1D convolution kernel
	double* B = (double*)malloc(size * sizeof(double));
	for (int i = 0; i < size; i++)
		B[i] = 1 / (sigma * sqrt(2.0 * 3.1415926)) * exp(-i * i / den);

	// normalize the 1D convolution kernel
	double norm = 0;
	for (int i = 0; i < size; i++)
		norm += B[i];
	norm *= 2;
	norm -= B[0];
	for (int i = 0; i < size; i++)
		B[i] /= norm;

	// convolution of each line of the input image
	double* R = (double*)malloc((size + xdim + size) * sizeof * R);
	
	// intialize the pointer of images
	float* IData = (float*)I.data;
	
	for (int k = 0; k < ydim; k++)
	{
		int i, j;
		for (i = size; i < bdx; i++)
			R[i] = *(IData + k * xdim + i - size);

		switch (boundary_condition)
		{
		case BOUNDARY_CONDITION_DIRICHLET:
			for (i = 0, j = bdx; i < size; i++, j++)
				R[i] = R[j] = 0;
			break;

		case BOUNDARY_CONDITION_REFLECTING:
			for (i = 0, j = bdx; i < size; i++, j++) {
				R[i] = *(IData + k * xdim + size - i);
				R[j] = *(IData + k * xdim + xdim - i - 1);
			}
			break;

		case BOUNDARY_CONDITION_PERIODIC:
			for (i = 0, j = bdx; i < size; i++, j++) {
				R[i] = *(IData + k * xdim + xdim - size + i);
				R[j] = *(IData + k * xdim + i);
			}
			break;
		}

		for (i = size; i < bdx; i++)
		{
			double sum = B[0] * R[i];
			for (j = 1; j < size; j++)
				sum += B[j] * (R[i - j] + R[i + j]);
			*(IData + k * xdim + i - size) = sum;
		}
	}

	// convolution of each column of the input image
	double* T = (double*)malloc((size + ydim + size) * sizeof * T);

	for (int k = 0; k < xdim; k++)
	{
		int i, j;
		for (i = size; i < bdy; i++)
			T[i] = *(IData + (i - size) * xdim + k);

		switch (boundary_condition)
		{
		case BOUNDARY_CONDITION_DIRICHLET:
			for (i = 0, j = bdy; i < size; i++, j++)
				T[i] = T[j] = 0;
			break;

		case BOUNDARY_CONDITION_REFLECTING:
			for (i = 0, j = bdy; i < size; i++, j++) {
				T[i] = *(IData + (size - i) * xdim + k);
				T[j] = *(IData + (ydim - i - 1) * xdim + k);
			}
			break;

		case BOUNDARY_CONDITION_PERIODIC:
			for (i = 0, j = bdx; i < size; i++, j++) {
				T[i] = *(IData + (ydim - size + i) * xdim + k);
				T[j] = *(IData + i * xdim + k);
			}
			break;
		}

		for (i = size; i < bdy; i++)
		{
			double sum = B[0] * T[i];
			for (j = 1; j < size; j++)
				sum += B[j] * (T[i - j] + T[i + j]);
			*(IData + (i - size) * xdim + k) = sum;
		}
	}

	free(R);
	free(T);
}

#endif // !MASK_C
