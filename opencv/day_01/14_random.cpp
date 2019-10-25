#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
// #define len(x) sizeof(x)/sizeof(x[0])
const int W=400;

static Scalar randomColor( RNG& rng )
{
    int icolor = (unsigned) rng;
    return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

int Displaying_Random_Text( Mat& image, string window_name, RNG rng )
{
      int lineType = 8;
      int NUMBER=1;
      int x_1=0,x_2=W,y_1=0,y_2=W;
      for ( int i = 1; i < NUMBER; i++ )
      {
        Point org;
        org.x = rng.uniform(x_1, x_2);
        org.y = rng.uniform(y_1, y_2);
        putText( image, "Testing text rendering", org, rng.uniform(0,8),
                 rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
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

int main(int ac, char** av)
{
    mycout<<"basic geometric drawing"<<endl;
    /** 必须知识
    RNG rng( 0xFFFFFFFF );
    rng.uniform(1, 10);
    */
    RNG rng( 0xFFFFFFFF );
    Mat M1=Mat::zeros(W,W,CV_8UC3);
    Mat M2=Mat::zeros(W,W,CV_8UC3);
    Displaying_Random_Text(M1,"test",rng);


    // 显示
    Mat imgs[]={M1,M2};
    string str[]={"src","dst"};
    showImg(imgs,str,2);

    return 0;
}

