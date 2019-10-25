#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/**filter*/
void filterImg(Mat&src,Mat&dst) // 有问题
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    CV_Assert(channels==3);
    int nRows = src.rows;
    int nCols = src.cols * channels;

    int i,j;
    uchar* p1,*p2,*p3,*output;
    for( i = 1; i < nRows-1; ++i)
    {
        // 一次读三行
        p1 = src.ptr<uchar>(i-1); // 获取每一行的数组，p为数组首地址
        p2 = src.ptr<uchar>(i);
        p3 = src.ptr<uchar>(i+1);
        output = dst.ptr<uchar>(i);
        for ( j = 1; j < nCols-1; j++)
        {
            // p[j-1] // B
            // p[j]  // G
            // p[j+1] // R
            output[j]=saturate_cast<uchar>( p1[j-1]*0+p1[j]*(-1)+p1[j+1]*0+
                                                                    p2[j-1]*(-1)+p2[j]*5+p2[j+1]*(-1)+
                                                                    p3[j-1]*0+p3[j]*(-1)+p3[j+1]*0);
        }
    }

}


void filterImg2(Mat&src,Mat&dst)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    CV_Assert(channels==3);
    int nRows = src.rows;
    int nCols = src.cols * channels;

    int i,j;
    uchar* p1,*p2,*p3,*output;
    for( i = 1; i < nRows-1; ++i)
    {
        // 一次读三行
        p1 = src.ptr<uchar>(i-1); // 获取每一行的数组，p为数组首地址
        p2 = src.ptr<uchar>(i);
        p3 = src.ptr<uchar>(i+1);
        output = dst.ptr<uchar>(i);
        for (int c=0;c<channels;c++)// 通道数(按通道处理)
        {
            for ( j = c+channels; j < nCols-channels; j=j+channels)
            {
                // p[j-1] // B
                // p[j]  // G
                // p[j+1] // R
//                output[j]=saturate_cast<uchar>( p1[j-1]*0+p1[j]*(-1)+p1[j+1]*0+
//                                                                        p2[j-1]*(-1)+p2[j]*5+p2[j+1]*(-1)+
//                                                                        p3[j-1]*0+p3[j]*(-1)+p3[j+1]*0);

                output[j]= saturate_cast<uchar>(p1[j-channels]*0+p1[j]*(-1)+p1[j+channels]*0+
                                                                       p2[j-channels]*(-1)+p2[j]*5+p2[j+channels]*(-1)+
                                                                       p3[j-channels]*0+p3[j]*(-1)+p3[j+channels]*0);
            }
        }

    }

}


void filterImg3(Mat&src,Mat&dst)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    CV_Assert(channels==3);
    int nRows = src.rows;
    int nCols = (src.cols-1) * channels;

    int i,j;
    uchar* p1,*p2,*p3,*output;
    for( i = 1; i < nRows-1; ++i)
    {
        // 一次读三行
        p1 = src.ptr<uchar>(i-1); // 获取每一行的数组，p为数组首地址
        p2 = src.ptr<uchar>(i);
        p3 = src.ptr<uchar>(i+1);
        output = dst.ptr<uchar>(i);

        for ( j = channels; j < nCols-channels; j++) // j=j+channels 只会处理B通道
        {
            // p[j-1] // B
            // p[j]  // G
            // p[j+1] // R

            output[j]= saturate_cast<uchar>(p1[j-channels]*0+p1[j]*(-1)+p1[j+channels]*0+
                                                                   p2[j-channels]*(-1)+p2[j]*5+p2[j+channels]*(-1)+
                                                                   p3[j-channels]*0+p3[j]*(-1)+p3[j+channels]*0);
        }

    }

}


void filterImg4(Mat&src,Mat&dst)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    CV_Assert(channels==3);
    int nRows = src.rows;
    // int nCols = (src.cols-1) * channels;
    int nCols = src.cols;

    // Vec3b intensity;
    for(int y = 1; y < nRows-1; ++y)
    {
        for(int x=1;x<nCols-1;++x)
        {
            for (int c=0;c<channels;c++)
            {
                dst.at<Vec3b>(y, x)[c]=saturate_cast<uchar>(src.at<Vec3b>(y-1, x)[c]*(-1)+src.at<Vec3b>(y+1, x)[c]*(-1)+
                src.at<Vec3b>(y, x)[c]*5+src.at<Vec3b>(y, x-1)[c]*(-1)+src.at<Vec3b>(y, x+1)[c]*(-1));
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
    Mat dst=Mat::zeros(A.size(),A.type());

    double t = getTickCount();
    // 使用自定义方法
    // filterImg(A,dst); // 有问题
    // filterImg2(A,dst);
    // filterImg3(A,dst);
    filterImg4(A,dst);

    /*
    // 使用内置方法
    // Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0); // 对比度增强
    Mat kernel = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); // sobel算子Gx 边缘检测
    // Mat kernel = (Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1); // sobel算子Gy 边缘检测
	filter2D(A, dst, A.depth(), kernel);
	*/

    double timeconsume = (getTickCount() - t) / getTickFrequency();
    printf("tim consume %.2f\n", timeconsume);

    // 显示
    Mat img[]={A,dst};
    string str[]={"origin","dst"};
    showImg(img,str,2);

    return 0;
}

