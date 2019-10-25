#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>
#include "common/book.h"

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/* 全局线程id get thread id: 1D block and 2D grid  <<<(32,32),32>>>*/
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)  // 2D grid,1D block
// #define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x+threadIdx.y*blockDim.x)  // 2D grid,2D block

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* 每个block的线程id*/
// #define get_tid_per_block() (threadIdx.x+threadIdx.y*blockDim.x) // 2D block
#define get_tid_per_block() (threadIdx.x)

using namespace std;
typedef float FLOAT;


__global__ void global_conv(FLOAT *dev_x,FLOAT *dev_z,FLOAT *dev_kernel,int height,int width,int kernel_size)
{
    /**block 是计算的基础单元*/
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    //int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    //int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    if(idx>=height*width) return; // 超出边界

    // 根据idx 反算出image的行与列
    int row=idx/width;
    int col=idx%width;

    FLOAT img_value=0;
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
                img_value=dev_x[cur_row*width+cur_col];
            }
            dev_z[idx]+=img_value*dev_kernel[j+i*kernel_size]; // 与对应的卷积核上的值相乘
        }
    }
}


int main()
{
    mycout<<"实现二维卷积"<<endl;
    const int height=1080; // image 高
    const int width=1920; // image 宽

    // 把二维展开成一维 （数组内存都是按一维排列）
    int nbytes=height*width*sizeof(FLOAT);

    FLOAT *host_x=NULL,*host_z=NULL;
    FLOAT *dev_x=NULL,*dev_z=NULL;
    int kernel_size=3;
    FLOAT *dev_kernel=NULL; // 卷积核
    FLOAT host_kernel[kernel_size*kernel_size]={0, -1, 0, -1, 5, -1, 0, -1, 0}; // 3x3 卷积核
    int kernelBytes=kernel_size*kernel_size*sizeof(FLOAT);

    /**1D block*/
    int bs=1024;

    /**1D grid*/
    int grid=ceil(1.0*height*width/bs);

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, nbytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_kernel, kernelBytes));

    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            host_x[j+i*width]=1;
        }
    }

    // CPU-->GPU
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_kernel, host_kernel, kernelBytes, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成

    // CPU 启动 GPU kernel计算
    {
        // 使用全局变量
        global_conv<<<grid,bs>>>(dev_x,dev_z,dev_kernel,height,width,kernel_size);

        // 使用共享内存变量 (共享内存变量只能block内部通信，跨block没法访问)
        // 卷积需要访问周边相邻的元素，如果卷积核左上角对应的像素和右下角对应的像素不在
        // 一个block内，这样没法通过共享内存来访问，因此必须使用全局变量
        // shared_conv<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_z,dev_kernel,height,width,kernel_size);
    }

    // GPU ---> CPU
    HANDLE_ERROR(cudaMemcpy(host_z, dev_z, nbytes, cudaMemcpyDeviceToHost));

    // 打印部分结果
    mycout<<"before conv"<<endl;
    for(int i=0;i<10;++i)
    {
        for(int j=0;j<10;++j)
        {
            cout << host_x[j+i*width] << " ";
        }
        cout << endl;
    }

    mycout<<"after conv"<<endl;
    for(int i=0;i<10;++i)
    {
        for(int j=0;j<10;++j)
        {
            cout << host_z[j+i*width] << " ";
        }
        cout << endl;
    }

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_z));
    HANDLE_ERROR(cudaFree(dev_kernel));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}