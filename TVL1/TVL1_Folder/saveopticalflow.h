#pragma once
#ifndef SAVEOPTICALFLOW_H
#define SAVEOPTICALFLOW_H

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//

#include <iostream>
#include <string>
#include <exception>
#include <fstream>
#include <opencv2/opencv.hpp>


// return whether flow vector is unknown
//bool unknown_flow(float u, float v)
//{
//	return (fabs(u) > UNKNOWN_FLOW_THRESH)
//		|| (fabs(v) > UNKNOWN_FLOW_THRESH)
//		|| isnan(u) || isnan(v);
//}
//
//
//bool unknown_flow(float* f)
//{
//	return unknown_flow(f[0], f[1]);
//}


void ReadFlowFile(cv::Mat& img, std::string filename)
{
	if (filename.size() == 0)
		throw "ReadFlowFile: empty filename";

	const std::string file_extension_true = "flo";
	const std::string file_extension = filename.substr(filename.size() - 3, filename.size());
	if (file_extension != file_extension_true)
		throw "ReadFlowFile: extension .flo expected";

	std::fstream fin(filename, std::ios::in | std::ios::binary);
	if (!fin)
		throw "ReadFlowFile: could not open";

	float tag = 0;
	int width = 0, height = 0;
	fin.read((char*)&tag, sizeof(float));
	fin.read((char*)&width, sizeof(int));
	fin.read((char*)&height, sizeof(int));

	if (tag != TAG_FLOAT) // simple test for correct endian-ness
		throw "ReadFlowFile: wrong tag (possibly due to big-endian machine?)";

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999)
		throw "ReadFlowFile: illegal width";

	if (height < 1 || height > 99999)
		throw "ReadFlowFile: illegal height";

	img = cv::Mat(cv::Size(width, height), CV_32FC2);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (!fin.eof())
			{
				float* img_pointer = img.ptr<float>(i, j);
				fin.read((char*)&img_pointer[0], sizeof(float));
				fin.read((char*)&img_pointer[1], sizeof(float));
			}
		}
	}
	fin.close();
}


void WriteFlowFile(cv::Mat& img, std::string filename)
{
	if (filename.size() == 0)
		throw "ReadFlowFile: empty filename";

	const std::string file_extension_true = "flo";
	const std::string file_extension = filename.substr(filename.size() - 3, filename.size());
	if (file_extension != file_extension_true)
		throw "ReadFlowFile: extension .flo expected";

	int width = img.cols;
	int height = img.rows;
	int nBands = img.channels();

	if (nBands != 2)
		throw "WriteFlowFile: image must have 2 bands";

	std::fstream fout(filename, std::ios::out | std::ios::binary);
	if (!fout)
		throw "WriteFlowFile: could not open";

	fout << TAG_STRING;
	fout.write((char*)&width, sizeof(int));
	fout.write((char*)&height, sizeof(int));

	// write the data
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float* img_pointer = img.ptr<float>(i, j);
			fout.write((char*)&img_pointer[0], sizeof(float));
			fout.write((char*)&img_pointer[1], sizeof(float));
		}
	}
	fout.close();
}

int mytest() {

	try
	{
		cv::Mat img = cv::Mat::ones(cv::Size(100, 100), CV_32FC2);
		std::string filename1 = "D:\\Code-VS\\picture\\FlowData\\flow.flo";
		std::string filename2 = "D:\\Code-VS\\picture\\FlowData\\flow1.flo";

		cv::Mat img1, img2;
		ReadFlowFile(img1, filename1);
		ReadFlowFile(img2, filename2);
		//WriteFlowFile(img, filename2);

		std::vector<cv::Mat> img1_split, img2_split;
		cv::split(img1, img1_split);
		cv::split(img2, img2_split);

		for (int i = 0; i < img1.rows; i++)
		{
			for (int j = 0; j < img1.cols; j++)
			{
				std::cout << i << "; " << j << std::endl;
				std::cout << *img1_split[0].ptr<float>(i, j) << "; " << *img2_split[0].ptr<float>(i, j) << std::endl;
			}
		}
	}
	catch (const char* exception)
	{
		std::cout << exception << std::endl;
		exit(1);
	}

	return 0;
}

#endif // !SAVEOPTICALFLOW_H
