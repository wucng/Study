#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

void drawByPixel(Mat& src)
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
            // 对角线直线为: y=(nRows/nCols)*x
            // if(y-x*nRows/nCols==0)
            if(y-x*(nRows-1)/(nCols-1)==0)
            {
                src.at<Vec3b>(y, x)[0]=255;
                src.at<Vec3b>(y, x)[1]=255;
                src.at<Vec3b>(y, x)[2]=255;
            }
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
    // Mat dst=Mat::zeros(A.size(),A.type());
    Mat dst=A.clone();

    double t = getTickCount();
    drawByPixel(dst); // 0.00161

    double timeconsume = (getTickCount() - t) / getTickFrequency();
    printf("tim consume: %.5f\n", timeconsume);

    // 显示
    Mat img[]={A,dst};
    string str[]={"origin","dst"};
    showImg(img,str,2);

    return 0;
}

