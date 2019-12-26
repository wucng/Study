/**
* https://blog.csdn.net/dcrmg/article/details/53543341
* g++ hello.cpp `pkg-config opencv --cflags --libs`
* 这里实现3种清晰度评价方法，分别是Tenengrad梯度方法、Laplacian梯度方法和方差方法。
* -------------------------------------------------------------------------------*
* 1.Tenengrad梯度方法
* Tenengrad梯度方法利用Sobel算子分别计算水平和垂直方向的梯度，同一场景下梯度值越高，图像越清晰。
* 以下是具体实现，这里衡量的指标是经过Sobel算子处理后的图像的平均灰度值，值越大，代表图像越清晰。
* -------------------------------------------------------------------------------*
* 2.Laplacian梯度方法：
* Laplacian梯度是另一种求图像梯度的方法，在上例的OpenCV代码中直接替换Sobel算子即可。
* -------------------------------------------------------------------------------*
* 3.方差方法：
* 方差是概率论中用来考察一组离散数据和其期望（即数据的均值）之间的离散（偏离）成都的度量方法。
* 方差较大，表示这一组数据之间的偏差就较大，组内的数据有的较大，有的较小，分布不均衡；
* 方差较小，表示这一组数据之间的偏差较小，组内的数据之间分布平均，大小相近。
* 对焦清晰的图像相比对焦模糊的图像，它的数据之间的灰度差异应该更大，即它的方差应该较大，
* 可以通过图像灰度数据的方差来衡量图像的清晰度，方差越大，表示清晰度越好。
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

int tenengrad_Sobel(Mat &imageSource,Mat &imageGrey)
{
	Mat imageSobel;
	Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
	
	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
	
	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Sobel Method): " + meanValueString;
	putText(imageSource, meanValueString, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
	// imshow("Articulation", imageSource);
	// waitKey();
	imwrite("../test2.jpg",imageSource);
	
	return 0;
}


int tenengrad_Laplacian(Mat &imageSource,Mat &imageGrey)
{
	Mat imageSobel;
	// Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
	Laplacian(imageGrey, imageSobel, CV_16U);
	
	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
	
	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Laplacian Method): " + meanValueString;
	putText(imageSource, meanValueString, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
	// imshow("Articulation", imageSource);
	// waitKey();
	imwrite("../test2.jpg",imageSource);
	
	return 0;
}


int variance_method(Mat &imageSource,Mat &imageGrey)
{
	Mat meanValueImage;
	Mat meanStdValueImage;
	
	//求灰度图像的标准差
	meanStdDev(imageGrey, meanValueImage, meanStdValueImage);
	
	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = meanStdValueImage.at<double>(0, 0);
	
	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Variance Method): " + meanValueString;
	putText(imageSource, meanValueString, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
	// imshow("Articulation", imageSource);
	// waitKey();
	imwrite("../test2.jpg",imageSource);
	
	return 0;
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
	
    //tenengrad_Sobel(srcImage,dstImage);
	tenengrad_Laplacian(srcImage,dstImage);
	//variance_method(srcImage,dstImage);

    return 0;
}