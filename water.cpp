#include <iostream>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2\core\types_c.h>
#include <opencv2\core\core_c.h>
#include <opencv2\imgproc\imgproc_c.h>
#include <opencv2\highgui\highgui_c.h>


using namespace std;
using namespace cv;



//全局函数声明
//Mat src, 
Mat dst;
int g_nTrackbarNmuer = 0;//0表示腐蚀，1表示膨胀  
int g_nStructElementSize = 4;//内核矩阵的尺寸      
Mat imreconstruct(Mat marker, Mat mask);
Mat diskStrel(int radius);
CvPoint centerPoint;
//Mat RegionGrow(Mat srcImage, Point pt);
//void on_mouse(int event, int x, int y, int flags, void* prarm);
void mouseHandler(int event, int x, int y, int flags, void* param);


void cv::erode(
    InputArray src,
    OutputArray dst,
    InputArray kernel,
    Point anchor,
    int iterations,
    int borderType,
    const Scalar& borderValue
);


void cv::dilate(
    InputArray src,
    OutputArray dst,
    InputArray kernel,
    Point anchor,
    int iterations,
    int borderType,
    const Scalar& borderValue
);

Mat diskStrel(int radius)
{
    Mat sel(2 * radius - 1, 2 * radius - 1, CV_8UC1, Scalar(1));
    /* the same as MATLAB function 'strel('disk',radius)' */
    int borderWidth = 0;
    switch (radius)
    {
    case 1: borderWidth = 0; break;
    case 3: borderWidth = 0; break;
    case 5: borderWidth = 2; break;
    case 7: borderWidth = 2; break;
    case 9: borderWidth = 4; break;
    case 11: borderWidth = 6; break;
    case 13: borderWidth = 6; break;
    case 15: borderWidth = 8; break;
    case 17: borderWidth = 8; break;
    case 19: borderWidth = 10; break;
    case 21: borderWidth = 10; break;
    default: borderWidth = 2; break;
    }

    for (int i = 0; i < borderWidth; i++) {
        for (int j = 0; j < borderWidth - i; j++) {
            sel.at<uchar>(i, j) = 0;
            sel.at<uchar>(i, sel.cols - 1 - j) = 0;
            sel.at<uchar>(sel.rows - 1 - i, j) = 0;
            sel.at<uchar>(sel.rows - 1 - i, sel.cols - 1 - j) = 0;
        }
    }

    return sel;
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
    do
    {
        dst.copyTo(temp1);
        dilate(dst, dst, Mat());
        cv::min(dst, mask, dst);
        compare(temp1, dst, temp2, CV_CMP_NE);
    } while (sum(temp2).val[0] != 0);
    return dst;
}



void mouseHandler(int event, int x, int y, int flags, void* param)
{
    IplImage* img0, * img1;
    img0 = (IplImage*)param;
    img1 = cvCreateImage(cvSize(img0->width, img0->height), img0->depth, img0->nChannels);
    cvCopy(img0, img1, NULL);

    CvFont font;
    uchar* ptr;
    char label[20];
    char label2[20];

    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 1);    //初始化字体

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        //读取像素
        ptr = cvPtr2D(img0, y, x, NULL);

        sprintf_s(label, "Grey Level:%d", ptr[0]);
        sprintf_s(label2, "Pixel: (%d, %d)", x, y);
        //调整显示位置
        if (img0->width - x <= 180 || img0->height - y <= 20)
        {
            cvRectangle(img1, cvPoint(x - 180, y - 40), cvPoint(x - 10, y - 10), CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
            cvPutText(img1, label, cvPoint(x - 180, y - 30), &font, CV_RGB(255, 255, 255));
            cvPutText(img1, label2, cvPoint(x - 180, y - 10), &font, CV_RGB(255, 255, 255));
        }
        else
        {
            cvRectangle(img1, cvPoint(x + 10, y - 12), cvPoint(x + 180, y + 20), CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
            cvPutText(img1, label, cvPoint(x + 10, y), &font, CV_RGB(255, 255, 255));
            cvPutText(img1, label2, cvPoint(x + 10, y + 20), &font, CV_RGB(255, 255, 255));
        }
        //以鼠标为中心画点

        centerPoint.x = x;
        centerPoint.y = y;
        cvCircle(img1, centerPoint, 3, CV_RGB(0, 0, 0), 1, 8, 3);


        cvShowImage("img", img1);
    }
}



int main(int argc, char** argv)
{

    Mat src = imread("1.jpg");

    if (src.empty())
    {
        std::cout << "Couldn't open file." << std::endl;
        system("pause");
        return -1;
    }
    int g_nStrutElement = 3;  //结构元素（内核矩阵）的尺寸
    Mat Iobr, Iobrd, J, pt;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
        Point(g_nStructElementSize, g_nStructElementSize));

    erode(src, dst, element);
    Iobr = imreconstruct(dst, src); //imshow("Image", dst);//开运算重构

    dilate(Iobr, Iobrd, element);
    Iobrd = ~Iobrd;
    Iobr = ~Iobr;
    cv::Mat Iobrcbr = imreconstruct(Iobrd, Iobr);
    dst = ~Iobrcbr;//基于重构的开闭运算
    //dst.convertTo(dst, CV_32F);
    //cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX);
    //dst.convertTo(dst, CV_32F);
    //Mat dst1 = Mat::zeros(dst.size(), CV_32FC1);
    //normalize(dst, dst1, 1.0, 0, NORM_MINMAX);
    //Mat result = dst1 * 255;
    //result.convertTo(dst1, CV_8UC1);
    //imshow("2--NORM_MINMAX", dst1);
    //imshow("IMAGE",dst);
    //waitKey();



    int exit = 0;
    int c;
    IplImage* img_dst = (IplImage*)&IplImage(dst);
    assert(img_dst);
    cvNamedWindow("img_dst", 1);
    cvSetMouseCallback("img_dst", mouseHandler, (void*)img_dst);
    cvShowImage("img", img_dst);
    cvWaitKey();
    while (!exit)
    {
        c = cvWaitKey(0);
        switch (c)
        {
        case 'q':
            exit = 1;
            break;
        default:
            continue;
        }
    }


    int x, y;
    x = centerPoint.x;
    y = centerPoint.y;
    //namedWindow("dst", CV_WINDOW_NORMAL);
    //选择种子点的代码
   //cvSetMouseCallback("dst", on_mouse, 0);
   // cvShowImage("src", dst);
    //WaitKey();
   // cv::imshow("pic", dst);

    //J = RegionGrow(dst, x,y);

    Mat growImage = Mat::zeros(dst.size(), CV_8UC1);   //创建一个空白区域，填充为黑色
   //double reg_mean = 0; //表示分割好的区域内的平均值，初始化为种子点的灰度值
    int reg_size = 1;//分割的到的区域，初始化只有种子点一个
    int neg_pos = 0; //用于记录neg_list中的待分析的像素点的个数
    double pixdist = 0;//记录最新像素点增加到分割区域后的距离测度,下一次待分析的四个邻域像素点和当前种子点的距离.如果当前坐标为（x, y）那么通过neigb我们可以得到其四个邻域像素的位置
    double dist, dist_min;
    int index;
    Point pToGrowing;                       //待生长点位置
    int neg_free = 90000;// 动态分配内存的时候每次申请的连续空间大小
    Mat neg_list = Mat::zeros(neg_free, 3, CV_8UC1);



    uchar* ptr = dst.ptr<uchar>(y, x);
    double reg_mean = 0;

    reg_mean = (int)*(ptr);//表示分割好的区域内的平均值，初始化为种子点的灰度值
    //reg_mean = dst.at<uchar>(y, x);
    int DIR[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };//四个邻域位置
    //index = neg_list.at<uchar>(neg_pos, 1);
    int width = dst.cols;
    int height = dst.rows;
    int numel = width * height;
    while (pixdist < 25 && reg_size < numel)
    {
        for (int j = 0; j < 4; j++)
        {
            pToGrowing.x = x + DIR[j][0];
            pToGrowing.y = y + DIR[j][1];
            int ins = (pToGrowing.x >= 0) && (pToGrowing.y >= 0) && (pToGrowing.x <= (dst.rows - 1)) && (pToGrowing.y <= (dst.cols - 1));
            if (ins && growImage.at<uchar>(pToGrowing.x, pToGrowing.y) == 0)
            {
                neg_pos = neg_pos + 1;
                neg_list.at<uchar>(neg_pos - 1, 0) = pToGrowing.x;
                neg_list.at<uchar>(neg_pos - 1, 1) = pToGrowing.y;
                neg_list.at<uchar>(neg_pos - 1, 2) = (int)*(dst.ptr<uchar>(pToGrowing.y, pToGrowing.x));// 存储对应点的灰度值
                growImage.at<uchar>(pToGrowing.x, pToGrowing.y) = 1;
            }
        }

        int z = neg_list.at<uchar>(0, 2);



        if (neg_pos + 10 > neg_free)
        {
            neg_free = neg_free + 100000;
            for (int i = neg_pos + 1; i <= neg_free; i++)
            {
                for (int j = 0; j < 3; j++)
                    neg_list.at<uchar>(i, j) = 0;
            }
        }
        //从所有待分析的像素点中选择一个像素点，该点的灰度值和已经分割好区域灰度均值的
        //差的绝对值时所待分析像素中最小的
        dist_min = abs((int)*(neg_list.ptr<uchar>(neg_pos, 2)) - reg_mean);
        for (int j = 0; j < neg_pos; ++j)
        {
            dist = abs((int)*(neg_list.ptr<uchar>(neg_pos, 2)) - reg_mean);
            if (dist < dist_min)
            {
                index = neg_pos;
                dist_min = dist;
            }
            else
                continue;
        }
        reg_size = reg_size + 1;
        reg_mean = (reg_mean * reg_size + neg_list.at<uchar>(index, 3)) / (reg_size);

        growImage.at<uchar>(x, y) = 2;

        x = neg_list.at<uchar>(index, 0);
        y = neg_list.at<uchar>(index, 1);

        neg_list.at<uchar>(index, 0) = neg_list.at<uchar>(neg_pos, 0);
        neg_list.at<uchar>(index, 1) = neg_list.at<uchar>(neg_pos, 1);
        neg_list.at<uchar>(index, 2) = neg_list.at<uchar>(neg_pos, 2);
        neg_pos = neg_pos - 1;
    }


    J = (J == 2);
    imshow("Image", J);






    waitKey();








    // J = RegionGrow(src, pt);
     //
     //imshow("Image", J);
     //waitKey();
}




