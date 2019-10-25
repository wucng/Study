#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define len(x) sizeof(x)/sizeof(x[0])

static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}

void gray2binary(const Mat &src,Mat &dst)
{
    // CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.type() == CV_8UC1);
    // int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;

    for(int y=0;y<nRows;y++)
    {
        for(int x=0;x<nCols;x++)
        {
            dst.at<uchar>(y,x)=saturate_cast<uchar>((src.at<uchar>(y,x)/128.0>1.0)?255:0);
        }
    }
    mycout<<"over"<<endl;
}

/**类似于tbb里的parallel_for使用*/
void gray2binary_parallel_for(const Mat &src,Mat &dst)
{
    // CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.type() == CV_8UC1);
    // int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;
    //*
    parallel_for_(Range(0, nRows*nCols), [&](const Range& range){
                  for (int r = range.start; r < range.end; r++)
                  {
                      int y = r / nCols; // 行
                      int x = r % nCols;  // 列
                      dst.at<uchar>(y,x)=saturate_cast<uchar>((src.at<uchar>(y,x)/128.0>1.0)?255:0);
                  }
                  });
    //*/
    /*
     parallel_for_(Range(0, nRows*nCols), [&](const Range& range){
                  uchar* p_src = src.data;// Mat对象的data数据成员将指针返回到第一行第一列。
                  uchar* p_dst = dst.data;// Mat对象的data数据成员将指针返回到第一行第一列。
                  for (int r = range.start; r < range.end; r++)
                  {
                      p_dst[r]=saturate_cast<uchar>((p_src[r]/128>=1)?255:0);
                  }
                  });
    */
    mycout<<"over"<<endl;
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

int main(int ac, char** av)
{
    mycout<<"parallel_for_ 并行实现灰度转二值图"<<endl;
    if (ac != 2)
    {
        help(av);
        return 1;
    }
    string filename = av[1];
    Mat src=imread(filename,IMREAD_GRAYSCALE );
    if(!src.data) // or  src.empty()
    {
        mycout<< "error"<<endl;
        return -1;
    }

    Mat dst=src.clone();

    // 计算时间
    double t = getTickCount();

    // gray2binary(src,dst); // 0.0061
    gray2binary_parallel_for(src,dst); // 0.0045

    double timeconsume = (getTickCount() - t) / getTickFrequency();
    printf("tim consume: %.5f\n", timeconsume);

    // 显示
    Mat imgs[]={src,dst};
    string str[]={"src","dst"};
    showImg(imgs,str,2);

    return 0;
}

