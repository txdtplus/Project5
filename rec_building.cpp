#include <rec_building_fun.h>

typedef struct Region
{                            
	int Area = 0;
	Mat PixelList;  // double CV_64F
	Mat Centroid;   // double CV_64F
	double MajorAxisLength;
	double MinorAxisLength;
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


int main(int argc, char* argv[])
{
	const char* src_name = "bwlabel_test.png";
	Mat src, L;
	Region* stats;
	int seg_num;
	
	/****************read source image************************/
	src = imread(src_name, IMREAD_UNCHANGED);
	if (src.empty()) 
	{
		fprintf(stderr, "Cannot load image %s\n", src_name);
		return -1;
	}


	bwlabel(src, L, &seg_num, 255);
	stats = regionprops(L, seg_num);

	for (int i = 0; i < 7; i++)
	{		
		cout << stats[i].Centroid <<"  " << stats[i].MinorAxisLength << endl;
	}

	return 0;
}

