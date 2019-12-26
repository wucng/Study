/**
g++ hello.cpp `pkg-config opencv --cflags --libs`

ԭ��˵���� ���ϳ��õ�һ�ַ����ǽ�RGBͼ��ת�䵽CIE Lab�ռ䣬
����L��ʾͼ�����ȣ�a��ʾͼ���/�̷�����b��ʾͼ���/��������
ͨ������ɫƫ��ͼ����a��b�����ϵľ�ֵ��ƫ��ԭ���Զ������Ҳ��ƫС��
ͨ������ͼ����a��b�����ϵľ�ֵ�ͷ���Ϳ�����ͼ���Ƿ����ɫƫ��
����CIE Lab*�ռ���һ���ȽϷ����Ĺ��̣�����OpenCV�ṩ���ֳɵĺ�����
�����������Ҳ�����ӡ�
��������������������������������
��Ȩ����������ΪCSDN���������ϵ��Ρ���ԭ�����£���ѭ CC 4.0 BY-SA ��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
ԭ�����ӣ�https://blog.csdn.net/qq_34997906/article/details/87970817
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
    // ͳ��a,b��Ӧͨ��������ֱ��ͼ
    float a=0.0f,b=0.0f;
    int *HistA = new int[256],*HistB = new int[256];
//    for(int i=0;i<256;i++)
//    {
//        HistA[i]=0;
//        HistB[i]=0;
//    }
    memset(HistA,0,sizeof(int)*256); // ��ʼ��Ϊ0
    memset(HistB,0,sizeof(int)*256);

    int height = dstImage.rows;
    int width = dstImage.cols;
    int channels = dstImage.channels();

    int x=0,y=0;

    unsigned char* dstPtr = dstImage.data; // ��ȡMat�洢��ַ

    for (int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            //�ڼ�������У�Ҫ���ǽ�CIEL*a*b*�ռ仹ԭ��ͬ
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
        //���㷶Χ-128��127
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

    mycout << "ɫƫָ����"<<cast<<endl;
    if (cast>1.1)
    {
        cout << "����ɫƫ" <<endl;
        return true;
    }
    else
    {
        cout << "������ɫƫ" <<endl;
        return false;
    }
}

int main(int argc, char *argv[])
{
    string path="../test.jpg";

    if (argc>1) // �ֶ��������
        path=argv[1];

    // load image
    Mat srcImage = imread(path,IMREAD_COLOR);
    if(srcImage.empty()) // or srcImage.data==NULL
    {
        mycout <<"image: "<<path<<" load fail"<<endl;
        return -1;
    }

    // Ŀ��ͼ��
    Mat dstImage = Mat::zeros(srcImage.size(),srcImage.depth());// ����һ����С������srcImageһ����ȫ0ͼ��
    //  ��RGBͼ��ת�䵽CIE L*a*b*
    cvtColor(srcImage,dstImage,COLOR_BGR2Lab);

    colorException(dstImage);

    return 0;
}
