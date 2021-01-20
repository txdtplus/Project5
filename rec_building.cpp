#include <rec_building_fun.h>


//int main(int argc, char* argv[])
//{
//	const char* src_name = "bwlabel_test.png";
//	Mat src, L;
//	Region* stats;
//	int seg_num;
//	
//	/****************read source image************************/
//	src = imread(src_name, IMREAD_UNCHANGED);
//	if (src.empty()) 
//	{
//		fprintf(stderr, "Cannot load image %s\n", src_name);
//		return -1;
//	}
//
//
//	bwlabel(src, L, &seg_num, 255);
//	stats = regionprops(L, seg_num);
//
//	for (int i = 0; i < 7; i++)
//	{		
//		cout << stats[i].Centroid <<"  " << stats[i].MinorAxisLength << endl;
//	}
//	/*Mat a, b;
//	a = Mat::zeros(1, 2, CV_64F);
//	b.create(1, 2, CV_64F);
//	cout << b << endl;
//
//	b.push_back(a);
//	b.push_back(a);
//
//	cout << b << endl;*/
//	return 0;
//}
#include<time.h>

int main()
{
	int num = 1e7;
	double a = 5;
	clock_t start, end;
	double runtime;

	start = clock();
	for (int i = 0; i < num; i++)
	{
		a = a * 1.0 + 2.0;
	}
	end = clock();
	runtime = (double(end) - double(start)) / CLOCKS_PER_SEC;

	printf("run time: %.5f", runtime);
}

//int main()
//{
//	cv::Mat src, dst;
//	cv::Mat SE;
//
//	src = imread("lenna.jpg");
//	//src.convertTo(src, CV_64F);
//	SE = getStructuringElement(CV_SHAPE_RECT, Size(5, 5));
//
//	clock_t start, end;
//	double runtime;
//
//	start = clock();
//	for (int i = 0; i < 50; i++)
//	{
//		dilate(src, dst, SE);
//		erode(dst, dst, SE);
//	}
//	end = clock();
//
//	namedWindow("dst", WINDOW_AUTOSIZE);
//	imshow("dst", dst);
//	waitKey(0);
//
//	runtime = (double(end) - double(start)) / CLOCKS_PER_SEC;
//
//	printf("run time: %.5f", runtime);
//}