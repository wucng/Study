#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/**改变图像的对比度和亮度*/
int change_contrast_brightness(const Mat&src,Mat&dst,double alpha=1.0,double beta =0.0)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    int nRows = src.rows;
    // int nCols = src.cols * channels;
    int nCols = src.cols; //* channels;

    Vec3b intensity_src;
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
            /*
            intensity_src=src.at<Vec3b>(y, x);
            intensity_dst=dst.at<Vec3b>(y, x);

            //
            intensity_dst.val[0]= saturate_cast<uchar>(intensity_src.val[0]*alpha+beta );
            intensity_dst.val[1]= saturate_cast<uchar>(intensity_src.val[1]*alpha+beta );
            intensity_dst.val[2]= saturate_cast<uchar>(intensity_src.val[2]*alpha+beta );

            // 修改像素值
            dst.at<Vec3b>(y, x)=intensity_dst;
            */


            intensity_src=src.at<Vec3b>(y, x);

            dst.at<Vec3b>(y, x)[0]= saturate_cast<uchar>(intensity_src.val[0]*alpha+beta );
            dst.at<Vec3b>(y, x)[1]= saturate_cast<uchar>(intensity_src.val[1]*alpha+beta );
            dst.at<Vec3b>(y, x)[2]= saturate_cast<uchar>(intensity_src.val[2]*alpha+beta );
        }
    }

    return 0;
}


int change_contrast_brightness2(const Mat&src,Mat&dst,double alpha=1.0,double beta =0.0)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;

    Vec3b intensity_src;
    Vec3b intensity_dst;

    for (int y=0;y<nRows;y++)
    {
        for (int x=0;x<nCols;x++)
        {
            for (int c=0;c<channels;c++)
            {
                dst.at<Vec3b>(y, x)[c]= saturate_cast<uchar>(alpha*src.at<Vec3b>(y, x)[c]+beta );
            }
        }
    }

    return 0;
}

int change_contrast_brightness3(Mat&src,Mat&dst,double alpha=1.0,double beta =0.0)
{
    CV_Assert(src.depth() == CV_8U);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols*channels;
    if (src.isContinuous())
    {
        nCols *= nRows; // 将[h,w*c] 转成一维 h*w*c
        nRows = 1;
    }

    uchar* p_src,*p_dst;

    for (int y=0;y<nRows;y++)
    {
        p_src = src.ptr<uchar>(y); // 获取每一行数组的首地址
        p_dst = dst.ptr<uchar>(y); // 获取每一行数组的首地址
        for (int x=0;x<nCols;x++)
        {
            dst.at<uchar>(y, x)=saturate_cast<uchar>(src.at<uchar>(y, x)*alpha+beta);// 0.03
            // p_dst[x]=saturate_cast<uchar>(p_src[x]*alpha+beta); // 0.02
            // *(p_dst+x)=saturate_cast<uchar>(*(p_src+x)*alpha+beta); // 0.03
            *p_dst++=saturate_cast<uchar>((*p_src++)*alpha+beta); // 0.02
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
    if (argc>1)
    {
        path=argv[1];
    }

    // 打开Image
    Mat img = imread(path, IMREAD_COLOR); // IMREAD_GRAYSCALE 以灰度图格式打开
    if (img.empty())
    {
        mycout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // 存放结果
    // Mat dst(img.size(),img.type(),Scalar(0));
    // Mat dst=Mat::zeros(img.size(),img.type());

    Mat dst=img.clone();
    dst=Scalar(0); //像素值全设为0

    // 计算时间
    double t = getTickCount();
    // change_contrast_brightness(img,dst); // 0.03
    // change_contrast_brightness2(img,dst); // 0.04
    change_contrast_brightness3(img,dst,1.5,5.0); // 0.02
    double timeconsume = (getTickCount() - t) / getTickFrequency();
    printf("tim consume %.2f\n", timeconsume);

    // 显示
    Mat imgs[]={img,dst};
    string str[]={"origin","dst"};
    showImg(imgs,str,2);

    return 0;
}

