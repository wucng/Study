#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"]"

/**传递函数指针做参数*/
Mat& test_cost_time(Mat& (*func)(Mat&,const uchar* const),Mat& I, const uchar* const table)
{
    double t = (double)getTickCount();
    // do something ...
    // Mat B=func(I,table);
    I=func(I,table);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;

    return I;
}

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U); // 必须是Uint8 数据类型
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols * channels;
    if (I.isContinuous())
    {
        nCols *= nRows; // 将[h,w*c] 转成一维 h*w*c
        nRows = 1;
    }
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i); // 获取每一行的地址
        for ( j = 0; j < nCols; ++j)
        {
            //p[j] = table[p[j]];
            // *(p+j)=table[*(p+j)];
            *p++ = table[*p];
        }
    }
    return I;
}

Mat& ScanImageByPointer(Mat &I,const uchar* const table)
{
    // 指针方式访问
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U); // 必须是Uint8 数据类型
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols * channels;
    if (I.isContinuous())
    {
        nCols *= nRows; // 将[h,w*c] 转成一维 h*w*c
        nRows = 1;
    }
    uchar* p = I.data;// Mat对象的data数据成员将指针返回到第一行第一列。
                                // 如果存储是连续的，我们可以使用它来遍历整个数据指针
    for( unsigned int i =0; i < nCols*nRows; ++i)
        *p++ = table[*p];

    return I;
}

Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
    // 迭代器访问
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            MatIterator_<uchar> it, end;
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                *it = table[*it];
            break;
        }
    case 3:
        {
            MatIterator_<Vec3b> it, end;
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
        }
    }
    return I;
}

Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            break;
        }
    case 3:
        {
         Mat_<Vec3b> _I = I;
         for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j )
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;
         break;
        }
    }
    return I;
}

// OpenCV提供了用于修改图像值的功能，而无需编写图像的扫描逻辑。
// 我们使用核心模块的cv :: LUT（）函数。首先，我们建立查找表的Mat类型：
Mat& CVLUT(Mat& I, const uchar* const table)
{
     Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = table[i];

    // LUT(I, lookUpTable, J); //I is our input image and J the output one
    LUT(I, lookUpTable, I);

    return I;
}

int main(int argc, char *argv[])
{
    if (argc<3) return -1;
    string path=argv[1];//"../data/test.jpg";

    int divideWith  = 0; //将输入字符串转换为数字-C ++样式
    stringstream s;
    s << argv [2];
    s >> divideWith; // 字符串转成int

    if(!s || !divideWith)
    {
        mycout << "Invalid number entered for dividing. " << endl;
        return -1;
    }

    uchar table[256]; // uchar  unsigned int8_t
    for (int i = 0; i < 256; ++i)
       table[i] = (uchar)(divideWith * (i/divideWith));

    // 打开Image
    Mat A = imread(path, IMREAD_COLOR); // here we'll know the method used (allocate matrix)
    if (A.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // 先复制一张原始的图，保证后面修改不会影响它
    Mat C=A.clone();

    // 测试
    Mat B;
    // B=test_cost_time(ScanImageAndReduceC,A,table); // 0.00500666
    // B=test_cost_time(ScanImageByPointer,A,table); // 0.00410707
    // B=test_cost_time(ScanImageAndReduceIterator,A,table); // 0.021447
    // B=test_cost_time(ScanImageAndReduceRandomAccess,A,table); // 0.0209611
    B=test_cost_time(CVLUT,A,table); // 0.0209611

    // 显示
    namedWindow( "origin", WINDOW_AUTOSIZE );
    imshow("origin", C);

    namedWindow( "deal with", WINDOW_AUTOSIZE );
    imshow("deal with", B);

    waitKey(0); // 等待用户退出

    return 0;
}

