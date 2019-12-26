/**
g++ hello.cpp `pkg-config opencv --cflags --libs`

原理说明： 网上常用的一种方法是将RGB图像转变到CIE Lab空间，
其中L表示图像亮度，a表示图像红/绿分量，b表示图像黄/蓝分量。
通常存在色偏的图像，在a和b分量上的均值会偏离原点很远，方差也会偏小；
通过计算图像在a和b分量上的均值和方差，就可评估图像是否存在色偏。
计算CIE Lab*空间是一个比较繁琐的过程，好在OpenCV提供了现成的函数，
因此整个过程也不复杂。
————————————————
版权声明：本文为CSDN博主「弦上的梦」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_34997906/article/details/87970817
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>

using namespace std;
using namespace cv;

#define mycout (cout<<" ["<< __FILE__ << " : "<<__LINE__<<"] "<<endl)

bool colorException(Mat& dstImage)
{
    // 统计a,b对应通道的像素直方图
    float a=0.0f,b=0.0f;
    int *HistA = new int[256],*HistB = new int[256];
//    for(int i=0;i<256;i++)
//    {
//        HistA[i]=0;
//        HistB[i]=0;
//    }
    memset(HistA,0,sizeof(int)*256); // 初始化为0
    memset(HistB,0,sizeof(int)*256);

    int height = dstImage.rows;
    int width = dstImage.cols;
    int channels = dstImage.channels();

    int x=0,y=0;

    unsigned char* dstPtr = dstImage.data; // 获取Mat存储地址

    for (int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            //在计算过程中，要考虑将CIEL*a*b*空间还原后同
            a += (float)(dstPtr[(i*width+j)*channels+1]-128);
            b += (float)(dstPtr[(i*width+j)*channels+2]-128);

            x = (int)dstPtr[(i*width+j)*channels+1];
            y = (int)dstPtr[(i*width+j)*channels+2];
            HistA[x]++;
            HistB[y]++;
        }
    }
    float  da=a/(height*width);
    float  db=b/(height*width);
    float D= (float)sqrt(da*da+db*db);
    float Ma=0.0f,Mb=0.0f;
    for(int i=0;i<256;i++)
    {
        //计算范围-128～127
        Ma+=abs(i-128-da)*HistA[i];
        Mb+=abs(i-128-db)*HistB[i];
    }
    Ma/=(float)(height*width);
    Mb/=(float)(height*width);
    float M=(float)sqrt(Ma*Ma+Mb*Mb);
    float K=D/M;
    float cast =K;

    // free
    delete[] HistA;
    delete[] HistB;

    mycout << "色偏指数："<<cast<<endl;
    if (cast>1.1)
    {
        cout << "存在色偏" <<endl;
        return true;
    }
    else
    {
        cout << "不存在色偏" <<endl;
        return false;
    }
}

int main(int argc, char *argv[])
{
    string path="../test.jpg";

    if (argc>1) // 手动输入参数
        path=argv[1];

    // load image
    Mat srcImage = imread(path,IMREAD_COLOR);
    if(srcImage.empty()) // or srcImage.data==NULL
    {
        mycout <<"image: "<<path<<" load fail"<<endl;
        return -1;
    }

    // 目标图像
    Mat dstImage = Mat::zeros(srcImage.size(),srcImage.depth());// 创建一个大小类型与srcImage一样的全0图像
    //  将RGB图像转变到CIE L*a*b*
    cvtColor(srcImage,dstImage,COLOR_BGR2Lab);

    colorException(dstImage);

    return 0;
}
