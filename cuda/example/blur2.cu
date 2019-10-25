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
#define image_size(image) (image.cols * image.rows)

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

__global__
void split_channel(unsigned char *dev_img,unsigned char *dev_B,
                   unsigned char *dev_G,unsigned char *dev_R,const int height,const int width)
{
    int idx=get_tid();
    if (idx>=height*width) return;
    // 根据idx 反算出image的行与列
    //int row=idx/width;
    //int col=idx%width;

    // opencv默认是BGR格式
    dev_B[idx]=dev_img[idx*3+0];
    dev_G[idx]=dev_img[idx*3+1];
    dev_R[idx]=dev_img[idx*3+2];
}


__global__
void  blur(unsigned char *dev_B,unsigned char *dev_B2,FLOAT *conv_kernel,
           const int height,const int width,const int kernel_size)
{
    int idx=get_tid();
    if (idx>=height*width) return;
    // 根据idx 反算出image的行与列
    int row=idx/width;
    int col=idx%width;

    unsigned char img_value=0;
    FLOAT tmp_value=0;
    int cur_row=0;
    int cur_col=0;

    for(int i=0;i<kernel_size;++i)
    {
        for(int j=0;j<kernel_size;++j)
        {
            // 找到卷积核左上角的对应的像素坐标
            cur_row=row-kernel_size/2+i;
            cur_col=col-kernel_size/2+j;
            if(cur_row<0 || cur_col<0 || cur_row>=height || cur_col>=width)
            {
                img_value=0;
            }
            else
            {
                // 反算对应的全局坐标
                img_value=dev_B[cur_row*width+cur_col];
            }
            tmp_value+=img_value*conv_kernel[j+i*kernel_size]; // 与对应的卷积核上的值相乘
        }
    }
    // dev_B2[idx]=(unsigned char)tmp_value; // 直接这么转有问题 有可能 负数变成 255
    dev_B2[idx]=FLOAT2uchar(tmp_value);
}


__global__
void concat_channel(unsigned char *dev_B2,unsigned char *dev_G2,unsigned char *dev_R2,
                    unsigned char *dev_img,const int height,const int width)
{
    int idx=get_tid();
    if (idx>=height*width) return;
    // 根据idx 反算出image的行与列
    // int row=idx/width;
    // int col=idx%width;

    // opencv默认是BGR格式
    dev_img[idx*3+0]=dev_B2[idx];
    dev_img[idx*3+1]=dev_G2[idx];
    dev_img[idx*3+2]=dev_R2[idx];
}



int main(int argc,char* argv[])
{
    mycout<<"卷积实现blur \n"<<
    "1、将BGR图像拆分成3个通道\n"<<
    "2、每个通道分别做卷积\n"<<
    "3、将3个通道合并得最终的结果\n"
    "输入格式: ./main xxxx.jpg xxxx.jpg"<<endl;

    if(argc<3) return -1;

    // 打开图片
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }

    // 卷积核
    const int kernel_size=3;
    FLOAT tmp_kernel[]={0,-1,0,-1,5,-1,0,-1,0};
    FLOAT *conv_kernel=NULL;
    // 使用统一内存 （同时被CPU与GPU访问）
    //HANDLE_ERROR(cudaMallocManaged((void**)&conv_kernel,kernel_size*kernel_size*sizeof(FLOAT),
    //                               cudaMemAttachGlobal));
    // conv_kernel=tmp_kernel;
	HANDLE_ERROR( cudaMalloc( (void**)&conv_kernel,kernel_size*kernel_size*sizeof(FLOAT) ) );
    HANDLE_ERROR( cudaMemcpy( conv_kernel,tmp_kernel,kernel_size*kernel_size*sizeof(FLOAT) ,cudaMemcpyHostToDevice ) );


    /**============使用GPU将图像拆分成3个通道============================*/
    Mat B=Mat::zeros(img.size(),CV_8UC1);
    Mat G=Mat::zeros(img.size(),CV_8UC1);
    Mat R=Mat::zeros(img.size(),CV_8UC1);

    // 分配GPU内存
    unsigned char *dev_img=NULL,*dev_B=NULL,*dev_G=NULL,*dev_R=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_img,image_size(img)*3 ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_B,image_size(img) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_G,image_size(img) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R,image_size(img) ) );

    HANDLE_ERROR( cudaMemcpy( dev_img,img.data,image_size(img)*3 ,cudaMemcpyHostToDevice ) );

    // 调用GPU核函数
    dim3 grid(img.rows,img.cols);
    split_channel<<<grid,1>>>(dev_img,dev_B,dev_G,dev_R,img.rows,img.cols);


    /**============3个通道分别做blur============================*/
    unsigned char *dev_B2=NULL,*dev_G2=NULL,*dev_R2=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_B2,image_size(img) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_G2,image_size(img) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R2,image_size(img) ) );

    // 调用GPU核函数
    blur<<<grid,1>>>(dev_B,dev_B2,conv_kernel,img.rows,img.cols,kernel_size);
    blur<<<grid,1>>>(dev_G,dev_G2,conv_kernel,img.rows,img.cols,kernel_size);
    blur<<<grid,1>>>(dev_R,dev_R2,conv_kernel,img.rows,img.cols,kernel_size);

    /**============使用3个通道合并============================*/
    unsigned char *dev_img2=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_img2,image_size(img)*3 ) );
    concat_channel<<<grid,1>>>(dev_B2,dev_G2,dev_R2,dev_img2,img.rows,img.cols);

    // GPU -->CPU
    // 创建一个空的Mat
    Mat blurImg=Mat::zeros(img.size(),CV_8UC3);

    HANDLE_ERROR( cudaMemcpy( blurImg.data,dev_img2,image_size(img)*3,cudaMemcpyDeviceToHost ) );

    // 保存结果
    imwrite(argv[2],blurImg);

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_img));
    HANDLE_ERROR(cudaFree(dev_B));
    HANDLE_ERROR(cudaFree(dev_G));
    HANDLE_ERROR(cudaFree(dev_R));

    HANDLE_ERROR(cudaFree(dev_img2));
    HANDLE_ERROR(cudaFree(dev_B2));
    HANDLE_ERROR(cudaFree(dev_G2));
    HANDLE_ERROR(cudaFree(dev_R2));

    HANDLE_ERROR(cudaFree(conv_kernel));

    return 0;
}
