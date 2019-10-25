#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

#define SAVE 0
#define SHOW 1

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    string path="../data/test.jpg";

    if (argc>1) // 手动输入参数
        path=argv[1];

    /**
    * IMREAD_UNCHANGED (<0) 表示加载原图，不做任何改变
    * IMREAD_GRAYSCALE ( 0)表示把原图作为灰度图像加载进来
    * IMREAD_COLOR (>0) 表示把原图作为RGB图像加载进来
    */
    Mat img = imread(path, IMREAD_COLOR);
    // if(img.empty()) // 加载失败 or  if(!img.data)
    if(!img.data)
   {
       cout<<"image: "<<path<<" load fail"<<endl;
       return -1;
   }

   #if SHOW
   /**
   * WINDOW_AUTOSIZE会自动根据图像大小，显示窗口大小，不能人为改变窗口大小
    * WINDOW_NORMAL,跟QT集成的时候会使用，允许修改窗口大小。
   */

   // 通道变换
   Mat grayImg;
   cvtColor( img, grayImg, COLOR_BGR2GRAY );// opencv 默认是BGR格式 而不是RGB格式


    namedWindow( "origin", WINDOW_AUTOSIZE );
    imshow("origin", img);

    namedWindow( "gray", WINDOW_AUTOSIZE );
    imshow("gray", grayImg);

    waitKey(0); // 等待用户退出

    #endif // SHOW

    #if  SAVE
    // 保存图片
    vector<int> compression_params; // 压缩质量
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(50); // 50% 质量  0~100
    /*
    PNG 图像是：
    compression_params.push_back(IMWRITE_PNG_COMPRESSION );
    compression_params.push_back(9); // 0~9
    */
    imwrite("./test.jpg",img,compression_params);
    // imwrite("./test.jpg",img);

    #endif // SAVE

    return 0;
}

