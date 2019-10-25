#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"]"

/**转灰度图*/
void BGR2Gray(Mat&I,Mat&dst)
{
    CV_Assert(I.depth() == CV_8U);
    int channels = I.channels();
    CV_Assert(channels==3);
    int nRows = I.rows;
    int nCols = I.cols * channels;

    int i,j;
    uchar* p, *output;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i); // 获取每一行的地址
        output = dst.ptr<uchar>(i);
        for ( j = 1; j < nCols-1; j=j+3)
        {
            // p[j-1] // B
            // p[j]  // G
            // p[j+1] // R

            // output[j/3]=(uchar)((p[j+1]*30+p[j]*59+p[j-1]*11+50)/100.0);
            output[j/3]=saturate_cast<uchar>(p[j+1]*0.299+p[j]*0.587+p[j-1]*0.114);
        }
    }

}

void BGR2Gray2(Mat&I,Mat&dst)
{
    CV_Assert(I.depth() == CV_8U);
    int channels = I.channels();
    CV_Assert(channels==3);
    int nRows = I.rows;
    int nCols = I.cols * channels;
    if (I.isContinuous())
    {
        nCols *= nRows; // 将[h,w*c] 转成一维 h*w*c
        nRows = 1;
    }

    int i,j;
    uchar *output=dst.data;
    uchar* p=I.data; // Mat对象的data数据成员将指针返回到第一行第一列。

    for(int j=1;j<nRows*nCols-1;j++)
    {
        // B :p[j-1]
        // G :p[j]
        // R :p[j+1]

        // output[j/3]=(uchar)((p[j+1]*30+p[j]*59+p[j-1]*11+50)/100.0);
        output[j/3]=saturate_cast<uchar>(p[j+1]*0.299+p[j]*0.587+p[j-1]*0.114);
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
    Mat gray=Mat::zeros(A.size(),CV_8UC1); // 灰度图只有一个通道

    BGR2Gray2(A,gray);

    // 显示
    Mat img[]={A,gray};
    string str[]={"origin","gray"};
    showImg(img,str,2);

    return 0;
}

