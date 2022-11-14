#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "bicubic_interpolation.h"
#include "mask.h"
#include "zoom.h"

#include "Myutils.h"
#include "flowshow.h"
#include "tvl1flow_lib.h"

#define PAR_DEFAULT_OUTFLOW "flow.flo"
#define PAR_DEFAULT_NPROC   0
#define PAR_DEFAULT_TAU     0.25
#define PAR_DEFAULT_LAMBDA  0.3
#define PAR_DEFAULT_THETA   0.3
#define PAR_DEFAULT_NSCALES 5
#define PAR_DEFAULT_ZFACTOR 0.5
#define PAR_DEFAULT_NWARPS  5
#define PAR_DEFAULT_EPSILON 0.01
#define PAR_DEFAULT_VERBOSE 0

/**
 *
 *  Main program:
 *   This program reads the following parameters from the console and
 *   then computes the optical flow:
 *   -nprocs      number of threads to use (OpenMP library)
 *   -I0          first image
 *   -I1          second image
 *   -tau         time step in the numerical scheme
 *   -lambda      data term weight parameter
 *   -theta       tightness parameter
 *   -nscales     number of scales in the pyramidal structure
 *   -zfactor     downsampling factor for creating the scales
 *   -nwarps      number of warps per scales
 *   -epsilon     stopping criterion threshold for the iterative process
 *   -out         name of the output flow field
 *   -verbose     switch on/off messages
 *
 */

int main(int argc, char* argv[])
{
	std::cout << "Version: " << CV_VERSION << std::endl;

	//read the parameters
	int i = 1;
	std::string outfile = (argc > i) ? argv[i] : PAR_DEFAULT_OUTFLOW;	i++;
	int   nproc = (argc > i) ? atoi(argv[i]) : PAR_DEFAULT_NPROC;   i++;
	float tau = (argc > i) ? atof(argv[i]) : PAR_DEFAULT_TAU;     i++;
	float lambda = (argc > i) ? atof(argv[i]) : PAR_DEFAULT_LAMBDA;  i++;
	float theta = (argc > i) ? atof(argv[i]) : PAR_DEFAULT_THETA;   i++;
	int   nscales = (argc > i) ? atoi(argv[i]) : PAR_DEFAULT_NSCALES; i++;
	float zfactor = (argc > i) ? atof(argv[i]) : PAR_DEFAULT_ZFACTOR; i++;
	int   nwarps = (argc > i) ? atoi(argv[i]) : PAR_DEFAULT_NWARPS;  i++;
	float epsilon = (argc > i) ? atof(argv[i]) : PAR_DEFAULT_EPSILON; i++;
	int   verbose = (argc > i) ? atoi(argv[i]) : PAR_DEFAULT_VERBOSE; i++;

	//check parameters
	if (nproc < 0) {
		nproc = PAR_DEFAULT_NPROC;
		if (verbose) {
			std::cout << stderr << "warning: nproc changed to " << nproc << std::endl;
		}
	}
	if (tau <= 0 || tau > 0.25) {
		tau = PAR_DEFAULT_TAU;
		if (verbose) {
			std::cout << stderr << "warning: tau changed to " << tau << std::endl;
		}
	}
	if (lambda <= 0) {
		lambda = PAR_DEFAULT_LAMBDA;
		if (verbose) {
			std::cout << stderr << "warning: lambda changed to " << lambda << std::endl;
		}
	}
	if (theta <= 0) {
		theta = PAR_DEFAULT_THETA;
		if (verbose) {
			std::cout << stderr << "warning: theta changed to " << theta << std::endl;
		}
	}
	if (nscales <= 0) {
		nscales = PAR_DEFAULT_NSCALES;
		if (verbose) {
			std::cout << stderr << "warning: nscales changed to " << nscales << std::endl;
		}
	}
	if (zfactor <= 0 || zfactor >= 1) {
		zfactor = PAR_DEFAULT_ZFACTOR;
		if (verbose) {
			std::cout << stderr << "warning: zfactor changed to " << zfactor << std::endl;
		}
	}
	if (nwarps <= 0) {
		nwarps = PAR_DEFAULT_NWARPS;
		if (verbose) {
			std::cout << stderr << "warning: nwarps changed to " << nwarps << std::endl;
		}
	}
	if (epsilon <= 0) {
		epsilon = PAR_DEFAULT_EPSILON;
		if (verbose) {
			std::cout << stderr << "warning: epsilon changed to " << epsilon << std::endl;
		}
	}

	verbose = 1;

	// read the input images
	std::string fixed_image_path = "E:\\Paper\\OpticalFlowData\\other-data\\Dimetrodon\\frame10.png";
	std::string moved_image_path = "E:\\Paper\\OpticalFlowData\\other-data\\Dimetrodon\\frame11.png";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);
	
	cv::cvtColor(fixed_image, fixed_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, moved_image, cv::COLOR_BGR2GRAY);

	cv::Mat fixed_image_32FC1, moved_image_32FC1;
	fixed_image.convertTo(fixed_image_32FC1, CV_32FC1);
	moved_image.convertTo(moved_image_32FC1, CV_32FC1);

	cv::Mat flow_x = cv::Mat::zeros(fixed_image.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(fixed_image.size(), CV_32FC1);

	std::cout << fixed_image.channels() << "; " << fixed_image.cols << "; " << fixed_image.rows << std::endl;
	std::cout << moved_image.channels() << "; " << moved_image.cols << "; " << moved_image.rows << std::endl;

	/*---------------------------------------Test the basic functions---------------------------------------*/
	//bicubic_interpolation_warp(fixed_image, flow_x, flow_y, moved_image, fixed_image.cols, fixed_image.rows, true);
	//divergence(flow_x, flow_y, moved_image, fixed_image.cols, fixed_image.rows);
	//forward_gradient(fixed_image, flow_x, flow_y, fixed_image.cols, fixed_image.rows);
	//centered_gradient(fixed_image, flow_x, flow_y, fixed_image.cols, fixed_image.rows);
	//gaussian(fixed_image, fixed_image.cols, fixed_image.rows, 2);
	//int nxx, nyy;
	//zoom_size(fixed_image.cols, fixed_image.rows, &nxx, &nyy, 0.5);
	//zoom_out(fixed_image, moved_image, fixed_image.cols, fixed_image.rows, 0.5);
	//zoom_in(fixed_image, moved_image, fixed_image.cols, fixed_image.rows, fixed_image.cols * 1.5, fixed_image.rows * 1.5);
	//std::cout << fixed_image.cols << "; " << moved_image.cols << std::endl;
	//std::cout << fixed_image.rows << "; " << moved_image.rows << std::endl;
	//fixed_image.convertTo(fixed_image, CV_8UC1);
	//cv::namedWindow("fixed_image2", cv::WINDOW_NORMAL);
	//cv::imshow("fixed_image2", fixed_image);

	int nx, ny, nx2, ny2;
	nx = fixed_image.cols;
	ny = fixed_image.rows;
	nx2 = moved_image.cols;
	ny2 = moved_image.rows;

	//read the images and compute the optical flow
	if (nx == nx2 && ny == ny2)
	{
		//Set the number of scales according to the size of the
		//images.  The value N is computed to assure that the smaller
		//images of the pyramid don't have a size smaller than 16x16
		const float N = 1 + log(hypot(nx, ny) / 16.0) / log(1 / zfactor);
		if (N < nscales)
			nscales = N;

		if (verbose) {
			std::cout << "nproc = " << nproc
				<< "; tau = " << tau
				<< "; lambda = " << lambda
				<< "; theta = " << theta << std::endl;
			std::cout << "nscales = " << nscales
				<< "; zfactor = " << zfactor
				<< "; nwarps =  " << nwarps
				<< "; epsilon = " << epsilon << std::endl;
		}

		//allocate memory for the flow
		cv::Mat I0 = fixed_image_32FC1.clone();
		cv::Mat I1 = moved_image_32FC1.clone();
		cv::Mat u1 = cv::Mat::zeros(fixed_image.size(), CV_32FC1);
		cv::Mat u2 = cv::Mat::zeros(moved_image.size(), CV_32FC1);

		//compute the optical flow
		Dual_TVL1_optic_flow_multiscale(
			I0, I1, u1, u2, nx, ny, tau, lambda, theta,
			nscales, zfactor, nwarps, epsilon, verbose
		);

		//// read the pixel value of optical flow
		//float* u1Data = (float*)u1.data;
		//const int step = u1.step[0] / u1.step[1];
		//for (int i = 0; i < u1.rows; i++)
		//	for (int j = 0; j < u1.cols; j++)
		//	{
		//		const int p = step * i + u1.channels() * j;
		//		std::cout << *(u1Data + p) << std::endl;
		//	}

		// 光流场处理
		std::vector<cv::Mat> optical_flow_merge;
		optical_flow_merge.push_back(u1);
		optical_flow_merge.push_back(u2);
		cv::Mat optical_flow_field;
		merge(optical_flow_merge, optical_flow_field);

		cv::Mat flowrgb(optical_flow_field.size(), CV_8UC3);
		flo2img(optical_flow_field, flowrgb);
		cv::namedWindow("optical_flow", cv::WINDOW_NORMAL);
		cv::imshow("optical_flow", flowrgb);
		//cv::imwrite("Dimetrodon_optical_flow.png", flowrgb);

		std::vector<cv::Mat> flow_spilit;
		cv::split(optical_flow_field, flow_spilit);

		// 通过后面一帧重建前面一帧
		cv::Mat result;
		//movepixels_2d2(moved_image, result, flow_spilit[0], flow_spilit[1], cv::INTER_CUBIC);
		movepixels_2d2(moved_image, result, u1, u2, cv::INTER_CUBIC);

		cv::namedWindow("result", cv::WINDOW_NORMAL);
		cv::imshow("result", result);
		cv::namedWindow("Res2", cv::WINDOW_NORMAL);
		cv::imshow("Res2", abs(result - fixed_image));
	}
	else {
		std::cout << "ERROR: input images size mismatch " << std::endl;
		return EXIT_FAILURE;
	}

	cv::waitKey();
	return EXIT_FAILURE;
}