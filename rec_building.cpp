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
	Mat src, dst;
	Mat visited;
	Mat locs_x, locs_y;
	Mat idx;
	Mat out_of_bounds;
	Mat stack, loc;
	int i, j;
	int ID_counter = 1;
	
	//cv::sort(idx, idx, SORT_ASCENDING);
	/****************read source image and convert to float image************************/
	src = imread(src_name);
	if (src.empty()) 
	{
		fprintf(stderr, "Cannot load image %s\n", src_name);
		return -1;
	}
	src = src / 255;
	visited = Mat::zeros(src.size(), CV_8U);
	dst = Mat::zeros(src.size(), CV_32S);

	for (int row = 0; row < 1; row++)
	{
		for (int col = 0; col < 1; col++)
		{
			cout << visited.at<uchar>(row, col) << endl;
			if (src.at<uchar>(row, col) == 0)
			{
				visited.at<uchar>(row, col) = 1;
			}
			else if(visited.at<uchar>(row, col))
			{
				continue;
			}
			else
			{
				stack = (cv::Mat_<int>(1, 2) << row, col);
				cout << !stack.empty() << endl << endl;
				while (!stack.empty())
				{
					loc = stack.row(stack.rows - 1).clone();
					stack.pop_back();

					i = loc.at<int>(0, 0);
					j = loc.at<int>(0, 1);
					if (visited.at<uchar>(i, j))
					{
						continue;
					}

					visited.at<uchar>(i, j) = 1;
					dst.at<int>(i, j) = ID_counter;

					locs_x = (cv::Mat_<int>(8, 1) << i - 1, i, i + 1, i - 1, i + 1, i - 1, i, i + 1);
					locs_y = (cv::Mat_<int>(8, 1) << j - 1, j - 1, j - 1, j, j, j + 1, j + 1, j + 1);
					
					out_of_bounds = locs_x < 0;
					cv::bitwise_or(out_of_bounds, locs_x >= src.rows, out_of_bounds);
					cv::bitwise_or(out_of_bounds, locs_y < 0, out_of_bounds);
					cv::bitwise_or(out_of_bounds, locs_y >= src.cols, out_of_bounds);
					cv::findNonZero(out_of_bounds, idx);

					cout << locs_x << endl;
					cout << locs_y << endl << endl;
					deleteRows(locs_x, idx, locs_x);
					deleteRows(locs_y, idx, locs_y);
					cout << locs_x << endl;
					cout << locs_y << endl;

				}
			}
		}
	}

	//loc_x = (cv::Mat_<int>(9, 1) << x - 1, x - 1, x - 3, x, x, x, x + 1, x + 1, x + 1);
	//loc_y = (cv::Mat_<int>(9, 1) << y - 1, y, y + 1, y - 1, y, y + 1, y - 1, y, y + 1);
	
	////idx.create(loc_x.rows, loc_x.cols, CV_32SC1);
	//cv::findNonZero(out_of_bounds, idx);
	//cout << loc_x << endl;
	//cout << loc_y << endl;
	//cout << idx << endl << endl;
	//deleteRows(loc_x, idx, loc_x);
	//deleteRows(loc_y, idx, loc_y);
	//cout << loc_x << endl;
	//cout << !src.empty() << endl;



	return 0;
}

