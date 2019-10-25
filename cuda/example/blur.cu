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
#define image_size(image) (image.cols * image.rows * image.channels())

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

__global__ void blur(unsigned char *dev_Img,unsigned char *dev_blurImg, FLOAT *conv_kernel,
                     int height,int width,int kernel_size)
{
    // map from blockIdx to pixel position
    int x = blockIdx.x; // cols
    int y = blockIdx.y; // rows
    int idx = x + y * gridDim.x;
    // if (idx>=height*width) return;

    // 根据idx 反算出image的行与列
    int row=idx/width;
    int col=idx%width;

    int cur_row=0;
    int cur_col=0;

    // 默认是BGR格式
    unsigned char B=0;
    unsigned char G=0;
    unsigned char R=0;

    FLOAT tmp_B=0;
    FLOAT tmp_G=0;
    FLOAT tmp_R=0;

	int offset=0;

    for(int i=0;i<kernel_size;++i)
    {
        for (int j=0;j<kernel_size;++j)
        {
            // 找到卷积核左上角的对应的像素坐标
            cur_row=row-kernel_size/2+i;
            cur_col=col-kernel_size/2+j;
            if(cur_row<0 || cur_col<0 || cur_row>=height || cur_col>=width)
            {
                B=0;G=0;R=0;
            }
            else
            {
				offset=cur_row*width+cur_col;
                B=dev_Img[offset*3+0];
                G=dev_Img[offset*3+1];
                R=dev_Img[offset*3+2];
            }

            tmp_B += B*conv_kernel[j+i*kernel_size]; // 与对应的卷积核上的值相乘
            tmp_G += G*conv_kernel[j+i*kernel_size];
            tmp_R += R*conv_kernel[j+i*kernel_size];
        }
    }
	
	// 直接转会导致 负数 变成255(结果有问题)
    //dev_blurImg[idx*3+0] = (unsigned char)tmp_B;
    //dev_blurImg[idx*3+1] = (unsigned char)tmp_G;
    //dev_blurImg[idx*3+2] = (unsigned char)tmp_R;
	
	//dev_blurImg[idx*3+0] = static_cast<unsigned char>(tmp_B);
    //dev_blurImg[idx*3+1] = static_cast<unsigned char>(tmp_G);
    // dev_blurImg[idx*3+2] = static_cast<unsigned char>(tmp_R);
	
	dev_blurImg[idx*3+0] = FLOAT2uchar(tmp_B);
    dev_blurImg[idx*3+1] = FLOAT2uchar(tmp_G);
    dev_blurImg[idx*3+2] = FLOAT2uchar(tmp_R);

}

int main(int argc,char* argv[])
{
    mycout<<"卷积实现blur \n"<<
    "输入格式: ./main xxxx.jpg xxxx.jpg"<<endl;

    if(argc<3) return -1;

    // 打开图片
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty())
    {
        mycout <<"load image fail"<<endl;
        return -1;
    }
    else
    {
        cout<<"height:"<<img.rows<<"\nwidth:"<<img.cols<<"\nchannels:"<<img.channels()<<endl;
    }

    // 卷积核
    const int kernel_size=3;
    FLOAT tmp_kernel[]={0,-1,0,-1,5,-1,0,-1,0};
    FLOAT *conv_kernel=NULL;
    // 使用统一内存 （同时被CPU与GPU访问）
    // HANDLE_ERROR(cudaMallocManaged((void**)&conv_kernel,kernel_size*kernel_size*sizeof(FLOAT),
     //                              cudaMemAttachGlobal));
    // conv_kernel=tmp_kernel;
    HANDLE_ERROR( cudaMalloc( (void**)&conv_kernel,kernel_size*kernel_size*sizeof(FLOAT) ) );
    HANDLE_ERROR( cudaMemcpy( conv_kernel,tmp_kernel,kernel_size*kernel_size*sizeof(FLOAT) ,cudaMemcpyHostToDevice ) );

    // 创建一个空的Mat
    // Mat blurImg=Mat::zeros(img.size(),CV_8UC3);
    Mat blurImg=Mat::zeros(img.size(),img.type());

    unsigned char *dev_Img; // GPU变量
    unsigned char *dev_blurImg; // GPU变量

    // GPU变量分配内存
    HANDLE_ERROR( cudaMalloc( (void**)&dev_Img,image_size(img)*sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_blurImg,image_size(blurImg)*sizeof(unsigned char) ) );
	HANDLE_ERROR(cudaMemset((void*)dev_blurImg,0,sizeof(unsigned char) * image_size(blurImg)));

    HANDLE_ERROR( cudaMemcpy( dev_Img,img.data,image_size(img) ,cudaMemcpyHostToDevice ) );

    // 启动核函数计算
    dim3 grid(img.rows,img.cols);
    blur<<<grid,1>>>(dev_Img,dev_blurImg,conv_kernel,img.rows,img.cols,kernel_size);

    HANDLE_ERROR( cudaMemcpy( blurImg.data,dev_blurImg,image_size(blurImg),cudaMemcpyDeviceToHost ) );

    // 保存结果
    imwrite(argv[2],blurImg);

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_Img));
    HANDLE_ERROR(cudaFree(dev_blurImg));
    HANDLE_ERROR(cudaFree(conv_kernel));

    return 0;
}
