#include <rec_building_fun.h>

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

int main(int argc, char* argv[])
{
	const char* src_name = "bwlabel_test.png";
	cv::Mat src, dst;

	cv::Mat loc_x, loc_y;
	int x = 1;
	int y = 1;
	loc_x = (cv::Mat_<int>(9, 1) << x - 1, x - 1, x - 3, x, x, x, x + 1, x + 1, x + 1);
	loc_y = (cv::Mat_<int>(9, 1) << y - 1, y, y + 1, y - 1, y, y + 1, y - 1, y, y + 1);

	Mat idx;
	Mat out_of_bounds;
	

	//cv::sort(idx, idx, SORT_ASCENDING);
	/****************read source image and convert to float image************************/
	src = imread(src_name);
	if (src.empty()) 
	{
		fprintf(stderr, "Cannot load image %s\n", src_name);
		return -1;
	}

	out_of_bounds = loc_x < 1;
	cv::bitwise_or(out_of_bounds, loc_x > src.rows, out_of_bounds);
	cv::bitwise_or(out_of_bounds, loc_y < 1, out_of_bounds);
	cv::bitwise_or(out_of_bounds, loc_y > src.cols, out_of_bounds);
	//idx.create(loc_x.rows, loc_x.cols, CV_32SC1);
	cv::findNonZero(out_of_bounds, idx);
	cout << loc_x << endl;
	cout << loc_y << endl;
	cout << idx << endl << endl;
	deleteRows(loc_x, idx, loc_x);
	deleteRows(loc_y, idx, loc_y);
	cout << loc_x << endl;
	cout << loc_y << endl;

	//int a=1; //需要删除的行  注意：需要删除的行要在message的范围内
	//for(int i=0;i<message.rows;i++)
	//{
	//	if(i!=a) //第i行不是需要删除的
	//	{
	//		dst.push_back(message.row(i)); //把message的第i行加到dst矩阵的后面
	//	}
	//}
	//message=dst.clone();



	return 0;
}

