/**
 * g++ hello.cpp `pkg-config opencv --cflags --libs`
 * 原理说明：计算图片在灰度图上的均值和方差，当存在亮度异常时，均值会偏离均值点（可以假设为128），方差也会偏小；
 * 通过计算灰度图的均值和方差，就可评估图像是否存在过曝光或曝光不足。
 *
*/
 
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"]"

/**
* opencv 检测图片亮度
* brightnessException 计算并返回一幅图像的色偏度以及，色偏方向
* cast 计算出的偏差值，小于1表示比较正常，大于1表示存在亮度异常；当cast异常时，da大于0表示过亮，da小于0表示过暗
* 返回值通过cast、da两个引用返回，无显式返回值
*/
int brightnessException(Mat &dstImage)
{
    float a=0.0f;
    int *Hist = new int[256];
    memset(Hist,0,sizeof(int)*256);
    int height = dstImage.rows;
    int width = dstImage.cols;
	
	unsigned char *dstPtr = dstImage.data;// 指向Mat的指针
	int x=0;
    for (int i=0;i< height;++i)
    {
        for (int j=0;j< width;++j)
        {
            //在计算过程中，考虑128为亮度均值点
			a += (float)(dstPtr[i*width+j]-128);
			x = (int)dstPtr[i*width+j];
			Hist[x]++;
        }
    }
	float da =  a/(height*width);
	mycout <<da<<endl;
	float D = abs(da);
	float Ma = 0.0f;
	for (int i=0;i< 256;++i)
	{
		Ma += abs(i-128-da)*Hist[i];
	}
	Ma /= (height*width);
	float M=abs(Ma);
	float K=D/M;
	float cast = K;
	mycout <<"亮度指数: " << cast <<endl;
	
	// free
    delete[] Hist;
	
	if (cast >= 1)
	{
		cout << "亮度：" << da <<endl;
		if (da >0)
		{
			cout << "过亮" <<endl;
			return 2;
		}
		else
		{
			cout << "过暗" <<endl;
			return 1;
		}
	}
	else
	{
		cout << "亮度：正常" <<endl;
		return 0;
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
    Mat dstImage = Mat::zeros(srcImage.size(),CV_8UC1);// 创建一个大小类型与srcImage一样的全0图像
    //  将RGB图像转变到灰度图
    cvtColor(srcImage,dstImage,COLOR_BGR2GRAY);

    brightnessException(dstImage);

    return 0;
}