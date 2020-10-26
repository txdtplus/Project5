#include <rec_building_fun.h>

typedef struct Region
{                            
	int Area;
	Mat PixelList;
};

Region* regionprops(InputArray _L, int seg_num)
{
	Region* region = new Region[seg_num];
	Mat L;
	Mat loc;
	Mat emptyMat;
	int value;
	L = _L.getMat();
	loc.create(1, 2, CV_32S);

	for (int n = 0; n < seg_num; n++)
	{
		region[n].Area = 0;
		//region[n].PixelList = emptyMat;
	}

	for (int row = 0; row < L.rows; row++)
	{
		for (int col = 0; col < L.cols; col++)
		{
			value = L.at<int>(row, col) - 1;

			if (value >= 0)
			{
				region[value].Area++;
				loc.at<int>(0, 0) = row;
				loc.at<int>(0, 1) = col;
				//cout << loc << endl;		
				region[value].PixelList.push_back(loc);
			}			
		}
	}
	return region;
}


int main(int argc, char* argv[])
{
	const char* src_name = "bwlabel_test.png";
	Mat src, dst;
	int seg_num;
	
	/****************read source image************************/
	src = imread(src_name, IMREAD_UNCHANGED);
	if (src.empty()) 
	{
		fprintf(stderr, "Cannot load image %s\n", src_name);
		return -1;
	}

	bwlabel(src, dst, &seg_num, 255);
	Region* stats = regionprops(dst, seg_num);

	Mat pixellist_0, pixellist_1, pixellist_2;
	stats[0].PixelList.copyTo(pixellist_0);
	stats[1].PixelList.copyTo(pixellist_1);
	stats[2].PixelList.copyTo(pixellist_2);

	pixellist_0 = pixellist_0 + 1;
	pixellist_1 = pixellist_1 + 1;
	pixellist_2 = pixellist_2 + 1;
	cout << stats[0].Area << endl;
	cout << stats[0].PixelList << endl;
	/*dst = dst * int(255.0 / seg_num);
	dst.convertTo(dst, CV_8U);
	imshow("output", dst);
	waitKey(0);*/

	return 0;
}

