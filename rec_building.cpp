#include <rec_building_fun.h>


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

