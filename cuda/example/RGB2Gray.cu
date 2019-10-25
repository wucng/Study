/**
! nvcc RGB2Gray.cu -o main `pkg-config opencv --cflags --libs`
! ldd mian // 查看缺失什么库
! cp -r /usr/local/lib/libopencv* /lib    // 将库直接拷到/lib
*/

#include <iostream>
#include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "common/book.h"
// #include "common/image.h"

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/* 全局线程id get thread id: 1D block and 2D grid  <<<(32,32),32>>>*/
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)  // 2D grid,1D block
// #define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x+threadIdx.y*blockDim.x)  // 2D grid,2D block

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* 每个block的线程id*/
// #define get_tid_per_block() (threadIdx.x+threadIdx.y*blockDim.x) // 2D block
#define get_tid_per_block() (threadIdx.x)

#define get_ptr(image) ((unsigned char*)image.data)
#define image_size(image) (image.cols * image.rows * 3)
using namespace std;
using namespace cv;
typedef float FLOAT;

__device__
unsigned char FLOAT2uchar(FLOAT value)
{
    if(value < 0)
        value = 0;
    else if(value > 255)
        value = 255;
    return (unsigned char)value;
    //return saturate_cast<unsigned char>(value);
}

__global__ void rgb2gray(unsigned char *dev_RGBImg,unsigned char *dev_gray)
{
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    unsigned char R=dev_RGBImg[offset*3+0];
    unsigned char G=dev_RGBImg[offset*3+1];
    unsigned char B=dev_RGBImg[offset*3+2];

    // dev_gray[offset]=(unsigned char) (R*0.299+G*0.587+B*0.114);
    dev_gray[offset]=FLOAT2uchar(R*0.299+G*0.587+B*0.114);
}

int main(int argc,char* argv[])
{
    mycout<<"RGB2Gray\n"<<
    "输入格式: ./main xxxx.jpg xxxx.jpg"<<endl;

    if(argc<3) return -1;

    // 打开图片
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // 创建一个空的Mat
    Mat gray=Mat::zeros(img.size(),CV_8UC1); // 灰度图只有一个通道

    // 从BGR-->RGB(CV_8UC3)
    Mat RGBImg;
    cvtColor( img, RGBImg, COLOR_BGR2RGB);// opencv 默认是BGR格式

    // 获取首元素地址(首行首元素地址)
    unsigned char* host_RGBImg=get_ptr(RGBImg); // unsigned char*  (uchar)
    unsigned char* host_gray=get_ptr(gray);

    unsigned char *dev_RGBImg; // GPU变量
    unsigned char *dev_gray; // GPU变量

    // GPU变量分配内存
    HANDLE_ERROR( cudaMalloc( (void**)&dev_RGBImg,image_size(RGBImg) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_gray,image_size(gray)/3 ) );

    HANDLE_ERROR( cudaMemcpy( dev_RGBImg,host_RGBImg,image_size(RGBImg) ,cudaMemcpyHostToDevice ) );

    // 启动核函数计算
    dim3 grid(RGBImg.rows,RGBImg.cols);
    rgb2gray<<<grid,1>>>(dev_RGBImg,dev_gray);

    HANDLE_ERROR( cudaMemcpy( host_gray,dev_gray,image_size(gray)/3,cudaMemcpyDeviceToHost ) );

    // 保存结果
    imwrite(argv[2],gray);

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_RGBImg));
    HANDLE_ERROR(cudaFree(dev_gray));

    return 0;
}
