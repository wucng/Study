#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    string path="../data/test.jpg";

    if (argc>1) // 手动输入参数
        path=argv[1];

    Mat A, C;                          // creates just the header parts
    A = imread(path, IMREAD_COLOR); // here we'll know the method used (allocate matrix)
    // 都是A的引用(指向A的地址)，任何一个修改了都会对其他有影响
    Mat B(A);                                 // Use the copy constructor
    C = A;                                    // Assignment operator

    Mat D (A, Rect(10, 10, 100, 100) ); // using a rectangle
    // Mat E = A(Range::all(), Range(1,3)); // using row and column boundaries
    Mat E = A(Range::all(), Range::all()); // 等价于 Mat E=A or Mat E(A);

    // 查看地址
    // A 的副本（拷贝），F,G修改不会影响A
    Mat F = A.clone();
    Mat G;
    A.copyTo(G);


    // namedWindow( "origin", WINDOW_AUTOSIZE );
    // imshow("origin", E);
    // waitKey(0);
    return 0;
}

