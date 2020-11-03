#pragma once
#include <iostream>
#include <opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <math.h>

#define pi 3.14159265358979323
#define MAX(A,B) A>B?A:B

using namespace std;
using namespace cv;

float maxVec3f(cv::Vec3f v);
float maxVec4f(cv::Vec4f v);
Mat MaxBand(cv::Mat image);
Mat intline(int x1, int x2, int y1, int y2);
Mat MakeLineStrel(float len, float theta_d);
Mat imreconstruct(Mat marker, Mat mask);
Mat cal_MBI(Mat src, double smin, double smax, double ds, int D);
Mat genColorResult(Mat src, Mat mask);
void bwlabel(InputArray _src, OutputArray _dst, int* seg_num, int max_value);
void deleteRows(InputArray _src, InputArray _idx, OutputArray _dst);
void MatPixel2Vec(InputArray _src, InputArray _locs_x, InputArray _locs_y, OutputArray _is_one);
void findNonZero_my(InputArray _src, OutputArray _idx);

void findNonZero_my(InputArray _src, OutputArray _idx)
{
	Mat src = _src.getMat();
	CV_Assert(src.channels() == 1 && src.dims == 2);

	int depth = src.depth();
	std::vector<Point> idxvec;
	int rows = src.rows, cols = src.cols;
	AutoBuffer<int> buf_(cols + 1);
	int* buf = buf_.data();

	for (int i = 0; i < rows; i++)
	{
		int j, k = 0;
		const uchar* ptr8 = src.ptr(i);
		if (depth == CV_8U || depth == CV_8S)
		{
			for (j = 0; j < cols; j++)
				if (ptr8[j] != 0) buf[k++] = j;
		}
		else if (depth == CV_16U || depth == CV_16S)
		{
			const ushort* ptr16 = (const ushort*)ptr8;
			for (j = 0; j < cols; j++)
				if (ptr16[j] != 0) buf[k++] = j;
		}
		else if (depth == CV_32S)
		{
			const int* ptr32s = (const int*)ptr8;
			for (j = 0; j < cols; j++)
				if (ptr32s[j] != 0) buf[k++] = j;
		}
		else if (depth == CV_32F)
		{
			const float* ptr32f = (const float*)ptr8;
			for (j = 0; j < cols; j++)
				if (ptr32f[j] != 0) buf[k++] = j;
		}
		else
		{
			const double* ptr64f = (const double*)ptr8;
			for (j = 0; j < cols; j++)
				if (ptr64f[j] != 0) buf[k++] = j;
		}

		if (k > 0)
		{
			size_t sz = idxvec.size();
			idxvec.resize(sz + k);
			for (j = 0; j < k; j++)
				idxvec[sz + j] = Point(buf[j], i);
		}
	}

	if (idxvec.empty() || (_idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous()))
		_idx.release();

	if (!idxvec.empty())
		Mat(idxvec).copyTo(_idx);
}

typedef struct Region
{
	int Area = 0;
	Mat PixelList;  // double CV_64F
	Mat Centroid;   // double CV_64F
	double MajorAxisLength = 0;
	double MinorAxisLength = 0;
};

Region* regionprops(InputArray _L, int seg_num)
{
	Region* region = new Region[seg_num];
	Mat L;
	Mat loc;
	int value;

	// Area and PixelList
	L = _L.getMat();
	loc.create(1, 2, CV_64F);
	for (int row = 0; row < L.rows; row++)
	{
		for (int col = 0; col < L.cols; col++)
		{
			value = L.at<int>(row, col) - 1;

			if (value >= 0)
			{
				region[value].Area++;
				loc.at<double>(0, 0) = col + 0.0;
				loc.at<double>(0, 1) = row + 0.0;

				region[value].PixelList.push_back(loc);
			}
		}
	}

	// Centroid
	for (int i = 0; i < seg_num; i++)
	{
		region[i].Centroid.create(1, 2, CV_64F);
		reduce(region[i].PixelList, region[i].Centroid, 0, CV_REDUCE_AVG);
	}

	// MajorAxisLength, MinorAxisLength
	Mat list;
	Mat x, y;
	Mat temp;

	double xbar, ybar;
	double uxx, uyy, uxy;
	double common;
	double Two_sqrt_2 = 2.0 * sqrt(2.0);

	for (int i = 0; i < seg_num; i++)
	{
		list = region[i].PixelList;
		xbar = region[i].Centroid.at<double>(0, 0);
		ybar = region[i].Centroid.at<double>(0, 1);

		x = list.col(0) - xbar;
		y = list.col(1) - ybar;

		uxx = cv::sum(x.mul(x))[0] / (x.rows) + 1.0 / 12.0;
		uyy = cv::sum(y.mul(y))[0] / (x.rows) + 1.0 / 12.0;
		uxy = cv::sum(x.mul(y))[0] / (x.rows);

		//cout << region[i].Centroid << "  " << ybar << "  " << uxy << endl;

		common = sqrt(pow((uxx - uyy), 2) + 4.0 * pow(uxy, 2));
		region[i].MajorAxisLength = Two_sqrt_2 * sqrt(uxx + uyy + common);
		region[i].MinorAxisLength = Two_sqrt_2 * sqrt(uxx + uyy - common);
	}

	return region;
}

void deleteRows(InputArray _src, InputArray _idx, OutputArray _dst)
{
	//delete some rows of src
	//idx denote the rows to be deleted
	Mat src, idx, dst;
	src = _src.getMat();
	idx = _idx.getMat();
	_dst.create(src.rows - idx.rows, src.cols, src.type());
	dst = _dst.getMat();

	int dst_i = 0;
	int idx_i = 0;
	for (int i = 0; i < src.rows; i++)
	{
		if (idx_i < idx.rows)
		{
			if (i != idx.at<Point>(idx_i, 0).y)
			{
				src.row(i).copyTo(dst.row(dst_i));
				dst_i++;
			}
			else
			{
				idx_i++;
			}
		}
		else
		{
			src.row(i).copyTo(dst.row(dst_i));
			dst_i++;
		}
	}
}


void MatPixel2Vec(InputArray _src, InputArray _locs_x, InputArray _locs_y, OutputArray _is_one)
{
	// Input: image, x, y locations.
	// x and y must be column vector.
	//
	// Output: the column vector of these pixels.
	Mat src, locs_x, locs_y;
	Mat is_one;
	src = _src.getMat();
	locs_x = _locs_x.getMat();
	locs_y = _locs_y.getMat();

	_is_one.create(locs_x.rows, 1, CV_8U);
	is_one = _is_one.getMat();

	int x, y;
	for (int i = 0; i < locs_x.rows; i++)
	{
		x = locs_x.at<int>(i, 0);
		y = locs_y.at<int>(i, 0);
		is_one.at<uchar>(i, 0) = src.at<uchar>(x, y);
	}
}

void bwlabel(InputArray _src, OutputArray _dst, int* seg_num, int max_value)
{
	// bwlabel function in MATLAB.
	// the input image must be 0 or max_value binary image .
	// the output Mat is labeled as 1, 2, 3, ...
	// the seg_num denotes the number of segmentations.
	Mat src, dst, zeros_mat;
	Mat visited;
	Mat locs_x, locs_y, locs;
	Mat idx;
	Mat out_of_bounds, is_visited, is_1;
	Mat stack, loc;
	int i, j;
	int ID_counter = 1;
	int idxx = 0;

	src = _src.getMat() / max_value;
	visited = Mat::zeros(src.size(), CV_8U);
	_dst.create(src.size(), CV_32S);
	dst = _dst.getMat();
	zeros_mat = Mat::zeros(src.size(), CV_32S);
	zeros_mat.copyTo(dst);

	// For each location in your matrix...
	for (int col = 0; col < src.cols; col++)
	{
		for (int row = 0; row < src.rows; row++)
		{
			// If this location is not 1, mark as visited and continue
			if (src.at<uchar>(row, col) == 0)
			{
				visited.at<uchar>(row, col) = 1;
			}
			// If we have visited, then continue
			else if (visited.at<uchar>(row, col))
			{
				continue;
			}
			// Else
			else
			{
				// Initialize your stack with this location
				stack = (cv::Mat_<int>(1, 2) << row, col);

				// While your stack isn't empty...
				while (!stack.empty())
				{
					loc = stack.row(stack.rows - 1).clone();

					// Pop off the stack
					stack.pop_back();

					i = loc.at<int>(0, 0);
					j = loc.at<int>(0, 1);

					// If we have visited this location, continue
					if (visited.at<uchar>(i, j))
					{
						continue;
					}

					// Mark location as true and mark this location to be its unique ID
					visited.at<uchar>(i, j) = 1;
					dst.at<int>(i, j) = ID_counter;

					//Look at the 8 neighbouring locations
					locs_x = (cv::Mat_<int>(9, 1) << i + 1, i, i - 1, i + 1, i, i - 1, i + 1, i, i - 1);
					locs_y = (cv::Mat_<int>(9, 1) << j + 1, j + 1, j + 1, j, j, j, j - 1, j - 1, j - 1);

					// Get rid of those locations out of bounds
					out_of_bounds = locs_x < 0;
					cv::bitwise_or(out_of_bounds, locs_x >= src.rows, out_of_bounds);
					cv::bitwise_or(out_of_bounds, locs_y < 0, out_of_bounds);
					cv::bitwise_or(out_of_bounds, locs_y >= src.cols, out_of_bounds);
					cv::findNonZero(out_of_bounds, idx);
					deleteRows(locs_x, idx, locs_x);
					deleteRows(locs_y, idx, locs_y);

					// Get rid of those locations already visited
					MatPixel2Vec(visited, locs_x, locs_y, is_visited);
					findNonZero(is_visited, idx);
					deleteRows(locs_x, idx, locs_x);
					deleteRows(locs_y, idx, locs_y);

					// Get rid of those locations that are zero.
					MatPixel2Vec(src, locs_x, locs_y, is_1);
					findNonZero(1 - is_1, idx);
					deleteRows(locs_x, idx, locs_x);
					deleteRows(locs_y, idx, locs_y);

					// Add remaining locations to the stack
					hconcat(locs_x, locs_y, locs);
					vconcat(locs, stack, stack);
				}

				// Increment counter once complete region has been examined
				ID_counter++;
			}
		}
	}

	*seg_num = ID_counter - 1;
}

Mat genColorResult(Mat src, Mat mask)
{
	//src: 源图像
	//mask: 二值图
	Mat dst;
	Mat normalize_mask;
	Mat inv_mask;

	cv::normalize(mask, normalize_mask, 1, 0, NORM_MINMAX);
	inv_mask = 1 - mask;

	if (src.channels() == 1)cvtColor(src, src, COLOR_GRAY2RGB);

	vector<Mat> channels;
	split(src, channels);
	channels[0] = channels[0].mul(inv_mask);
	channels[1].setTo(255, mask);
	channels[2].setTo(255, mask);
	merge(channels, dst);

	return dst;
}

float maxVec3f(cv::Vec3f v)
{
	//calculate maximum value for 3 bands
	float v_max;
	v_max = v[0] > v[1] ? v[0] : v[1];
	v_max = v_max > v[2] ? v_max : v[2];
	return v_max;
}


float maxVec4f(cv::Vec4f v)
{
	//calculate maximum value for 4 bands
	float v_max;
	v_max = v[0] > v[1] ? v[0] : v[1];
	v_max = v_max > v[2] ? v_max : v[2];
	v_max = v_max > v[3] ? v_max : v[3];
	return v_max;
}


Mat MaxBand(cv::Mat image_in)
{
	//input: a image
	//output: the maximum value of each band for every pixel
	//only support float Mat data
	cv::Mat image_out(image_in.size(), CV_32FC1);;
	cv::Vec3f pixel3f;
	cv::Vec4f pixel4f;

	switch (image_in.channels())
	{
	case 1:
		image_out = image_in;
		break;
	case 3:

		for (int i = 0; i < image_in.size().height; i++)
		{
			for (int j = 0; j < image_in.size().width; j++)
			{
				pixel3f = image_in.at<Vec3f>(i, j);
				image_out.at<float>(i, j) = maxVec3f(pixel3f);
			}
		}
		break;
	case 4:
		for (int i = 0; i < image_in.size().height; i++)
		{
			for (int j = 0; j < image_in.size().width; j++)
			{
				pixel4f = image_in.at<Vec4f>(i, j);
				image_out.at<float>(i, j) = maxVec4f(pixel4f);
			}

		}
		break;
	default:
		printf("this channels of image is not supportted\n");
		break;
	}
	return image_out;
}


Mat intline(int x1, int x2, int y1, int y2)
{
	/*the same function as intline in MATLAB*/
	int dx, dy, max_dxdy;
	int flipFlag = 0;
	int t;
	float m;
	Mat x, y, xy;

	dx = abs(x2 - x1);
	dy = abs(y2 - y1);

	max_dxdy = MAX(dx, dy);

	//rows:1, cols:max_dxdy + 1, 与函数说明相反，以实验结果为准
	x = Mat::zeros(1, max_dxdy + 1, CV_32S);
	y = Mat::zeros(1, max_dxdy + 1, CV_32S);
	xy = Mat::zeros(2, max_dxdy + 1, CV_32S);

	if (dx == 0 && dy == 0)
	{
		xy.at<int>(0, 0) = x1;
		xy.at<int>(1, 0) = y1;

		//cout << xy << endl;
		return xy;
	}

	if (dx >= dy)
	{
		if (x1 > x2)
		{
			t = x1; x1 = x2; x2 = t;
			t = y1; y1 = y2; y2 = t;
			flipFlag = 1;
		}
		m = float((y2 - y1)) / float((x2 - x1));
		for (int i = 0; i < dx + 1; i++)
		{
			x.at<int>(0, i) = x1 + i;
			y.at<int>(0, i) = round(y1 + m * (x.at<int>(0, i) - x1));
		}
	}
	else
	{
		if (y1 > y2)
		{
			t = x1; x1 = x2; x2 = t;
			t = y1; y1 = y2; y2 = t;
			flipFlag = 1;
		}
		m = float((x2 - x1)) / float((y2 - y1));
		for (int i = 0; i < dy + 1; i++)
		{
			y.at<int>(0, i) = y1 + i;
			x.at<int>(0, i) = round(x1 + m * (y.at<int>(0, i) - y1));
		}
	}

	if (flipFlag)
	{
		cv::flip(x, x, 1);//实验表明，这是左右翻转
		cv::flip(y, y, 1);
	}

	x.row(0).copyTo(xy.row(0));
	y.row(0).copyTo(xy.row(1));

	//cout << xy << endl;
	return xy;
}


Mat MakeLineStrel(float len, float theta_d)
{
	/* the same as MATLAB function 'strel('line',len,theta_d)' */
	double theta;
	int x, y, M, N;
	int col, row;
	Mat colrow;
	Mat LineSE;

	theta = int(theta_d) % 180 * pi / 180;
	x = round((len - 1.0) / 2.0 * cos(theta));
	y = -round((len - 1.0) / 2.0 * sin(theta));
	//矩阵列数增加的方向与纵坐标的方向相反

	M = 2 * abs(y) + 1;
	N = 2 * abs(x) + 1;


	colrow = intline(-x, x, -y, y);
	LineSE = Mat::zeros(M, N, CV_8U);

	for (int i = 0; i < colrow.cols; i++)
	{
		col = colrow.at<int>(0, i) + abs(x);
		row = colrow.at<int>(1, i) + abs(y);
		LineSE.at<uchar>(row, col) = 1;
	}

	return LineSE;
}


Mat imreconstruct(Mat marker, Mat mask)
{
	/*the same as MATLAB function imreconstruct*/
	Mat dst;
	marker.copyTo(dst);

	dilate(dst, dst, Mat());
	cv::min(dst, mask, dst);
	Mat temp1 = Mat(marker.size(), CV_32FC1);
	Mat temp2 = Mat(marker.size(), CV_32FC1);
	int iter = 0;
	do
	{
		dst.copyTo(temp1);
		dilate(dst, dst, Mat());
		cv::min(dst, mask, dst);
		compare(temp1, dst, temp2, CV_CMP_NE);
		iter++;
		if (iter == 5)break;
	} while (sum(temp2).val[0] != 0);
	return dst;
}


Mat cal_MBI(Mat src, double smin, double smax, double ds, int D)
{
	cv::Mat src_float, dst_float;
	cv::Vec3f pixel;
	cv::Vec3f pixel3f;
	cv::Mat LineSE;
	cv::Mat b, be, gamma, WTH;
	cv::Mat DMP, Sum_S_DMP, MP1, MP2;
	cv::Mat MBI, binMBI;

	double theta_d, len;
	double d_theta_d;
	int S;

	S = int((smax - smin) / ds + 1);

	/******************************convert to float image********************************/
	src.convertTo(src_float, CV_32FC3);
	b = MaxBand(src_float);
	cv::imwrite("C://zkyfile//matlab_program//self_learning//b.png", b);


	/*****************parameter setting**************************************************/
	MBI = Mat::zeros(b.size(), b.type());
	WTH = Mat::zeros(b.size(), b.type());
	theta_d = 0.0;
	d_theta_d = 180.0 / (D - 1.0);


	/**************************************caculation************************************/
	for (int i = 0; i < D; i++)
	{
		cout << "calculating direction... the " << i + 1 << "th loop" << endl;
		Sum_S_DMP = Mat::zeros(b.size(), b.type());
		len = smin + 0.0;
		b.copyTo(MP1);
		for (int j = 0; j < S; j++)
		{
			cout << "calculating length... the " << j + 1 << "th loop" << endl;
			LineSE = MakeLineStrel(len, theta_d);
			erode(b, be, LineSE);
			gamma = imreconstruct(be, b);

			WTH = b - gamma;
			WTH.copyTo(MP2);
			//MP2 = WTH;

			absdiff(MP2, MP1, DMP);
			//DMP = abs(MP2 - MP1);

			Sum_S_DMP = Sum_S_DMP + DMP;
			//MP1 = MP2;

			MP2.copyTo(MP1);
			len = len + ds;
		}
		MBI = MBI + Sum_S_DMP;
		theta_d = theta_d + d_theta_d;
	}

	MBI = MBI / (double(D + 0.0) * double(S + 0.0));


	/******************************display and imwrite************************************/
	//cv::imwrite("C://zkyfile//matlab_program//self_learning//MBI.png", MBI);
	cv::normalize(MBI, MBI, 255.0, 0.0, NORM_MINMAX);

	MBI.convertTo(MBI, CV_8UC1);
	double thresh = cv::threshold(MBI, binMBI, 0.0, 255.0, CV_THRESH_OTSU);
	cv::threshold(MBI, binMBI, thresh, 255.0, CV_THRESH_BINARY);
	binMBI.convertTo(binMBI, CV_8UC1);

	return binMBI;
}
