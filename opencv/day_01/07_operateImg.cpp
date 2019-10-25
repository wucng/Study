#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"]"


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
    Mat img = imread(path, IMREAD_COLOR); // IMREAD_GRAYSCALE 以灰度图格式打开
    if (img.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // imwrite("test.jpg", img); // 保存
    // 使用cv :: imdecode和cv :: imencode 内存中读写文件
    // 获取像素值
    int x=50,y(50);
    Scalar intensity = img.at<uchar>(y, x); // for CV_8UC1
    Scalar intensity2 = img.at<uchar>(Point(x, y));// for CV_8UC1

    // CV_8UC3 ，3通道
    Vec3b intensity = img.at<Vec3b>(y, x);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];

    // 对浮点图像
    Vec3f intensity = img.at<Vec3f>(y, x);
    float blue = intensity.val[0];
    float green = intensity.val[1];
    float red = intensity.val[2];

    //更改像素值：
    img.at<uchar>(y, x) = 128;

    // 原始操作
    img = Scalar(0); //像素值全设为0

    // 设置某行像素值
    img.row(i).setTo(Scalar(0,0,0));

    // 选择感兴趣的区域
    Rect r(10, 10, 100, 100);
    Mat smallImg = img(r);

    // 转灰度图
    cvtColor(img, grey, COLOR_BGR2GRAY);

    // 改变图像类型 Change image type from 8UC1 to 32FC1:
    src.convertTo(dst, CV_32F);

    //Visualizing images

    mycout<<intensity<<intensity2<<endl;

    // 显示
//    Mat img[]={img,dst};
//    string str[]={"origin","dst"};
//    showImg(img,str,2);

    return 0;
}

