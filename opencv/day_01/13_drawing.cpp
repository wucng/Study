#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
// #define len(x) sizeof(x)/sizeof(x[0])
const int W=400;

void drawRect(Mat& src,const Point& pt1,const Point& pt2,
              const Scalar & color=Scalar(0,0,255),int thickness = 1, //	thickness为负数表示填充
              int lineType = LINE_8,int 	shift = 0 )
{
    rectangle(src,pt1,pt2,color,thickness,lineType,shift);
}

void drawEllipse(Mat &src,const Point &center,Size axes,double 	angle,
                 double 	startAngle,double 	endAngle,const Scalar & 	color,
                 int 	thickness = 1,int lineType = LINE_8,int 	shift = 0)
{
    ellipse(src,center,axes,angle,startAngle,endAngle,color,thickness,lineType,shift);
}

void drawCircle(Mat& src,Point 	center,int 	radius,const Scalar & 	color,
                int 	thickness = 1,	int 	lineType = LINE_8,int 	shift = 0 )
{
    circle(src,center,radius,color,thickness,lineType,shift);
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
    Point pt; // 它表示一个2D点，由其图像坐标x和y指定
    pt.x = 10;
    pt.y = 8;
    // or Point pt =  Point(10, 8);

    Scalar( a, b, c );//  Blue = a, Green = b and Red = c
    */
    Mat M1=Mat::zeros( W, W, CV_8UC3 );
    Mat M2=Mat::zeros( W, W, CV_8UC3 );

    // drawRect(M1,Point(W/4, W/4),Point(3*W/4, 3*W/4));
    // drawRect(M2,Point(W/4, W/4),Point(3*W/4, 3*W/4),Scalar(255,0,0),-1);// 填充

    // drawEllipse(M1,Point(W/2, W/2),Size( W/4, W/16 ),90,0,360,Scalar(255,0,0));
    // drawEllipse(M2,Point(W/2, W/2),Size( W/4, W/16 ),90,0,360,Scalar(255,0,0),-1);// 填充

    // 画线
    line(M1,Point(0,0),Point(W-1,W-1),Scalar(0,0,255),2,LINE_AA );

    // 画箭头线
    arrowedLine(M2,Point(W/2,W/2),Point(W/4,W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(3*W/4,3*W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(3*W/4,W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(W/4,3*W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(W/2,3*W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(W/2,W/4),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(3*W/4,W/2),Scalar(0,0,255),2);
    arrowedLine(M2,Point(W/2,W/2),Point(W/4,W/2),Scalar(0,0,255),2);


    // 画多边形
    Point rook_points[1][20];
      rook_points[0][0]  = Point(    W/4,   7*W/8 );
      rook_points[0][1]  = Point(  3*W/4,   7*W/8 );
      rook_points[0][2]  = Point(  3*W/4,  13*W/16 );
      rook_points[0][3]  = Point( 11*W/16, 13*W/16 );
      rook_points[0][4]  = Point( 19*W/32,  3*W/8 );
      rook_points[0][5]  = Point(  3*W/4,   3*W/8 );
      rook_points[0][6]  = Point(  3*W/4,     W/8 );
      rook_points[0][7]  = Point( 26*W/40,    W/8 );
      rook_points[0][8]  = Point( 26*W/40,    W/4 );
      rook_points[0][9]  = Point( 22*W/40,    W/4 );
      rook_points[0][10] = Point( 22*W/40,    W/8 );
      rook_points[0][11] = Point( 18*W/40,    W/8 );
      rook_points[0][12] = Point( 18*W/40,    W/4 );
      rook_points[0][13] = Point( 14*W/40,    W/4 );
      rook_points[0][14] = Point( 14*W/40,    W/8 );
      rook_points[0][15] = Point(    W/4,     W/8 );
      rook_points[0][16] = Point(    W/4,   3*W/8 );
      rook_points[0][17] = Point( 13*W/32,  3*W/8 );
      rook_points[0][18] = Point(  5*W/16, 13*W/16 );
      rook_points[0][19] = Point(    W/4,  13*W/16 );
      const Point* ppt[1] = { rook_points[0] };
       int npt[] = { 20 };
    // polylines(M1,ppt,npt,1,true,Scalar(0,255,0),2);  // 第5个参数是false 不闭合
    fillPoly( M1,ppt,npt,1,Scalar( 255, 255, 255 ),LINE_AA );// 填充多边形

    // 显示
    Mat imgs[]={M1,M2};
    string str[]={"src","dst"};
    showImg(imgs,str,2);

    return 0;
}

