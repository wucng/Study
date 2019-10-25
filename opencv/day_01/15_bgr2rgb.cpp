#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

void BGR2RGB(Mat& src,Mat& dst)
{
    // CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.type() == CV_8UC3);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols; // * channels;

    for(int y=0;y<nRows;y++)
    {
        for (int x=0;x<nCols;x++)
        {
            // B , G ,R 顺序调换
            dst.at<Vec3b>(y, x)[0]=src.at<Vec3b>(y, x)[2];
            dst.at<Vec3b>(y, x)[1]=src.at<Vec3b>(y, x)[1];
            dst.at<Vec3b>(y, x)[2]=src.at<Vec3b>(y, x)[0];
        }
    }

}

void BGR2RGB2(Mat& src,Mat& dst)
{
    // CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.type() == CV_8UC3);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols * channels;

    for(int y=0;y<nRows;y++)
    {
        for (int x=0;x<nCols-channels+1;x+=channels)
        {
            // B , G ,R 顺序调换
            dst.at<uchar>(y, x)=src.at<uchar>(y, x+2);
            dst.at<uchar>(y, x+1)=src.at<uchar>(y, x+1);
            dst.at<uchar>(y, x+2)=src.at<uchar>(y, x);
        }
    }

}

void BGR2RGB3(Mat& src,Mat& dst)
{
    // CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.type() == CV_8UC3);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols * channels;

    uchar* p_src,*p_dst;
    for(int y=0;y<nRows;y++)
    {
        p_src=src.ptr<uchar>(y); // 某一行数组的首元素地址
        p_dst=dst.ptr<uchar>(y);
        for (int x=0;x<nCols-channels+1;x+=channels)
        {
            // B , G ,R 顺序调换
            p_dst[x]=p_src[x+2];
            p_dst[x+1]=p_src[x+1];
            p_dst[x+2]=p_src[x];
        }
    }

}

void showImg(Mat img[],string str[],int n)
{
    for (int i=0;i<n;i++)
    {
        namedWindow( str[i], WINDOW_AUTOSIZE );
        imshow(str[i], img[i]);
    }

    waitKey(0); // 等待用户退出
}

int main(int argc, char *argv[])
{
    string path="../data/test.jpg";
    if (argc>2)
        path=argv[1];

    // 打开Image
    Mat A = imread(path, IMREAD_COLOR); // here we'll know the method used (allocate matrix)
    if (A.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // 先复制一张原始的图，保证后面修改不会影响它
    // Mat C=A.clone();
    // 创建一个空的Mat
    // Mat dst=Mat::zeros(A.size(),CV_8UC3);
    Mat dst=Mat::zeros(A.size(),A.type());

    double t = getTickCount();
    // BGR2RGB(A,dst); // 0.018
    // BGR2RGB2(A,dst); // 0.013
    BGR2RGB3(A,dst); // 0.003

    double timeconsume = (getTickCount() - t) / getTickFrequency();
    printf("tim consume: %.5f\n", timeconsume);

    // 显示
    Mat img[]={A,dst};
    string str[]={"origin","dst"};
    showImg(img,str,2);

    return 0;
}
