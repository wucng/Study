#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    string path="../data/test.jpg";

    if (argc>1) // 手动输入参数
        path=argv[1];

    Mat A = imread(path, IMREAD_COLOR);
    Mat M=Mat::zeros(A.size(),A.type()); // 创建一个大小类型与A一样的全0图像
    // Mat M=Mat::zeros(A.size(),A.depth()); // 创建一个大小类型与A一样的全0图像
    // Mat M=Mat::zeros(A.rows,A.cols,CV_8UC(A.channels())); // 创建一个大小类型与A一样的全0图像
    // Mat M=Mat::zeros(A.size(),CV_8UC3); // 创建一个大小与A一样,类型为3通道Uint8的全0图像
    // Mat M(A.size(),CV_8UC3,Scalar(0,0,0)); // Scalar(0,0,0) 表示3个通道值全为0
    // Mat M(A.size(),CV_8UC3,Scalar(0)); // Scalar(0) 表示所有通道值全为0
    // Mat M(A.size(),CV_8UC1,Scalar::all(0)); // CV_8UC1 ,8U 表示UInt8 ,C1 表示1个通道
    // Mat::eye()
    // Mat::ones()
    cout <<"M.size:"<< M.size()<<"\n"<<
    "M.rows:"<< M.rows<<"\n"<<
    "M.cols:"<< M.cols<<"\n"<<
    "M.channels:"<< M.channels()<<"\n"<<
    "M.depth:"<< M.depth()<<"\n"<<
    "M.type:"<< M.type()<<endl;

    namedWindow( "origin", WINDOW_AUTOSIZE );
    imshow("origin", M);
    waitKey(0);
    return 0;
}

