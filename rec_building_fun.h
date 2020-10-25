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
