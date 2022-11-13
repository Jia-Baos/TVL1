#ifndef BICUBIC_INTERPOLATION_C
#define BICUBIC_INTERPOLATION_C

#include <iostream>
#include <opencv2/opencv.hpp>

#define BOUNDARY_CONDITION 0
//0 Neumann
//1 Periodic
//2 Symmetric

/**
  *
  * Neumann boundary condition test
  *
**/
static int neumann_bc(int x, int nx, bool* out)
{
	if (x < 0)
	{
		x = 0;
		*out = true;
	}
	else if (x >= nx)
	{
		x = nx - 1;
		*out = true;
	}

	return x;
}

/**
  *
  * Periodic boundary condition test
  *
**/
static int periodic_bc(int x, int nx, bool* out)
{
	if (x < 0)
	{
		const int n = 1 - (int)(x / (nx + 1));
		const int ixx = x + n * nx;

		x = ixx % nx;
		*out = true;
	}
	else if (x >= nx)
	{
		x = x % nx;
		*out = true;
	}

	return x;
}


/**
  *
  * Symmetric boundary condition test
  *
**/
static int symmetric_bc(int x, int nx, bool* out)
{
	if (x < 0)
	{
		const int borde = nx - 1;
		const int xx = -x;
		const int n = (int)(xx / borde) % 2;

		if (n) x = borde - (xx % borde);
		else x = xx % borde;
		*out = true;
	}
	else if (x >= nx)
	{
		const int borde = nx - 1;
		const int n = (int)(x / borde) % 2;

		if (n) x = borde - (x % borde);
		else x = x % borde;
		*out = true;
	}

	return x;
}


/**
  *
  * Cubic interpolation in one dimension
  *
**/
static double cubic_interpolation_cell(
	double v[4],  //interpolation points
	double x      //point to be interpolated
)
{
	return  v[1] + 0.5 * x * (v[2] - v[0] +
		x * (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] +
			x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}


/**
  *
  * Bicubic interpolation in two dimensions
  *
**/
static double bicubic_interpolation_cell(
	double p[4][4], //array containing the interpolation points
	double x,       //x position to be interpolated
	double y        //y position to be interpolated
)
{
	double v[4];
	v[0] = cubic_interpolation_cell(p[0], x);
	v[1] = cubic_interpolation_cell(p[1], x);
	v[2] = cubic_interpolation_cell(p[2], x);
	v[3] = cubic_interpolation_cell(p[3], x);
	return cubic_interpolation_cell(v, y);
}

/**
  *
  * Compute the bicubic interpolation of a point in an image.
  * Detect if the point goes outside the image domain.
  *
**/
float bicubic_interpolation_at(
	const cv::Mat input,//image to be interpolated
	const float	 uu,    //x component of the vector field
	const float  vv,    //y component of the vector field
	const int    nx,    //image width
	const int    ny,    //image height
	bool         border_out //if true, return zero outside the region
)
{
	int sx = (uu < 0) ? -1 : 1;
	int sy = (vv < 0) ? -1 : 1;

	int x, y, mx, my, dx, dy, ddx, ddy;
	bool out[1] = { false };

	//apply the corresponding boundary conditions
	switch (BOUNDARY_CONDITION) {

		// int prosess deletes the decimal places direactly
	case 0: x = neumann_bc((int)uu, nx, out);
		y = neumann_bc((int)vv, ny, out);
		mx = neumann_bc((int)uu - sx, nx, out);
		my = neumann_bc((int)vv - sy, ny, out);
		dx = neumann_bc((int)uu + sx, nx, out);
		dy = neumann_bc((int)vv + sy, ny, out);
		ddx = neumann_bc((int)uu + 2 * sx, nx, out);
		ddy = neumann_bc((int)vv + 2 * sy, ny, out);
		break;

	case 1: x = periodic_bc((int)uu, nx, out);
		y = periodic_bc((int)vv, ny, out);
		mx = periodic_bc((int)uu - sx, nx, out);
		my = periodic_bc((int)vv - sy, ny, out);
		dx = periodic_bc((int)uu + sx, nx, out);
		dy = periodic_bc((int)vv + sy, ny, out);
		ddx = periodic_bc((int)uu + 2 * sx, nx, out);
		ddy = periodic_bc((int)vv + 2 * sy, ny, out);
		break;

	case 2: x = symmetric_bc((int)uu, nx, out);
		y = symmetric_bc((int)vv, ny, out);
		mx = symmetric_bc((int)uu - sx, nx, out);
		my = symmetric_bc((int)vv - sy, ny, out);
		dx = symmetric_bc((int)uu + sx, nx, out);
		dy = symmetric_bc((int)vv + sy, ny, out);
		ddx = symmetric_bc((int)uu + 2 * sx, nx, out);
		ddy = symmetric_bc((int)vv + 2 * sy, ny, out);
		break;

	default:x = neumann_bc((int)uu, nx, out);
		y = neumann_bc((int)vv, ny, out);
		mx = neumann_bc((int)uu - sx, nx, out);
		my = neumann_bc((int)vv - sy, ny, out);
		dx = neumann_bc((int)uu + sx, nx, out);
		dy = neumann_bc((int)vv + sy, ny, out);
		ddx = neumann_bc((int)uu + 2 * sx, nx, out);
		ddy = neumann_bc((int)vv + 2 * sy, ny, out);
		break;
	}

	if (*out && border_out)
		return 0.0;

	else
	{
		float* inputData = (float*)input.data;

		const int step = input.step[0] / sizeof(inputData[0]);

		//obtain the interpolation points of the image
		const float p11 = *(inputData + input.channels() * mx + step * my);
		const float p12 = *(inputData + input.channels() * x + step * my);
		const float p13 = *(inputData + input.channels() * dx + step * my);
		const float p14 = *(inputData + input.channels() * ddx + step * my);

		const float p21 = *(inputData + input.channels() * mx + step * y);
		const float p22 = *(inputData + input.channels() * x + step * y);
		const float p23 = *(inputData + input.channels() * dx + step * y);
		const float p24 = *(inputData + input.channels() * ddx + step * y);

		const float p31 = *(inputData + input.channels() * mx + step * dy);
		const float p32 = *(inputData + input.channels() * x + step * dy);
		const float p33 = *(inputData + input.channels() * dx + step * dy);
		const float p34 = *(inputData + input.channels() * ddx + step * dy);

		const float p41 = *(inputData + input.channels() * mx + step * ddy);
		const float p42 = *(inputData + input.channels() * x + step * ddy);
		const float p43 = *(inputData + input.channels() * dx + step * ddy);
		const float p44 = *(inputData + input.channels() * ddx + step * ddy);

		//create array
		double pol[4][4] = {
			{p11, p12, p13, p14},
			{p21, p22, p23, p24},
			{p31, p32, p33, p34},
			{p41, p42, p43, p44}
		};

		//return interpolation
		return bicubic_interpolation_cell(pol, uu - x, vv - y);
	}
}


/**
  *
  * Compute the bicubic interpolation of an image.
  *
**/
void bicubic_interpolation_warp(
	const cv::Mat input,	// image to be warped
	const cv::Mat u,		// x component of the vector field
	const cv::Mat v,		// y component of the vector field
	cv::Mat& output,		// image warped with bicubic interpolation
	const int    nx,        // image width
	const int    ny,        // image height
	bool         border_out // if true, put zeros outside the region
)
{
	output = cv::Mat::zeros(input.size(), CV_32FC1);
	float* uData = (float*)u.data;
	float* vData = (float*)v.data;
	float* outputData = (float*)output.data;

	const int step = input.step[0] / sizeof(outputData[0]);

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			const int p = step * i + input.channels() * j;

			// the remaped position of poxels
			const float uu = (float)(j + *(uData + p));
			const float vv = (float)(i + *(vData + p));

			// obtain the bicubic interpolation at position (uu, vv)
			*(outputData + p) = bicubic_interpolation_at(input,
				uu, vv, nx, ny, border_out);
		}
	}
}

#endif // !BICUBIC_INTERPOLATION_C
