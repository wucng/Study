#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/**两张图像相加*/
int twoImgAdd(Mat&src1,Mat&src2,Mat&dst,double alpha=0.5,double gama=0.0)
{
    if(src1.type()!=src2.type() || src1.size()!=src2.size())
    {
        mycout<<"两张图片不匹配"<<endl;
        return -1;
    }
    CV_Assert(src1.depth() == CV_8U);
    int channels = src1.channels();
    int nRows = src1.rows;
    // int nCols = src1.cols * channels;
    int nCols = src1.cols ;//* channels;

    Vec3b intensity_src1;
    Vec3b intensity_src2;
    Vec3b intensity_dst;
    for (int y=0;y<nRows;y++)
    {
        for (int x=0;x<nCols;x++)
        {
            /*
            Vec3b intensity = img.at<Vec3b>(y, x);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            */
            intensity_src1=src1.at<Vec3b>(y, x);
            intensity_src2=src2.at<Vec3b>(y, x);
            intensity_dst=dst.at<Vec3b>(y, x);

            //
            intensity_dst.val[0]= saturate_cast<uchar>(intensity_src1.val[0]*alpha+intensity_src2.val[0]*(1-alpha)+gama);
            intensity_dst.val[1]=saturate_cast<uchar>(intensity_src1.val[1]*alpha+intensity_src2.val[1]*(1-alpha)+gama);
            intensity_dst.val[2]=saturate_cast<uchar>(intensity_src1.val[2]*alpha+intensity_src2.val[2]*(1-alpha)+gama);

            // 修改像素值
            dst.at<Vec3b>(y, x)=intensity_dst;
        }
    }

    return 0;
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
    string path2="../data/1.jpg";
    if (argc>3)
    {
        path=argv[1];
        path2=argv[2];
    }


    // 打开Image
    Mat img = imread(path, IMREAD_COLOR); // IMREAD_GRAYSCALE 以灰度图格式打开
    if (img.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    Mat img2 = imread(path2, IMREAD_COLOR); // IMREAD_GRAYSCALE 以灰度图格式打开
    if (img2.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // resize 到统一大小
    Mat img_resize=Mat::zeros(img2.size(),img2.type());
    resize(img,img_resize,img_resize.size());

    // 逐像素相加
    Mat dst=Mat::zeros(img2.size(),img2.type());

    twoImgAdd(img_resize,img2,dst);

    // 显示
    Mat imgs[]={img,dst};
    string str[]={"origin","dst"};
    showImg(imgs,str,2);

    return 0;
}

