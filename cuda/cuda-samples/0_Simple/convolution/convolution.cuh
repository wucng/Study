#include "common.h"

typedef float FLOAT;

template <typename T>
__global__ void global_conv(T *dev_x,T *dev_z,T *dev_kernel,int height,int width,int kernel_size)
{
    /**block 是计算的基础单元 <<<2,256>>>*/
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

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

// 使用共享内存 （推荐）
template <int SIZE>
__global__ void shared_conv(FLOAT *dev_x,FLOAT *dev_z,FLOAT *dev_kernel,int height,int width,int kernel_size)
{
    __shared__ FLOAT imgData[SIZE];
    __shared__ FLOAT wData[SIZE];
    __shared__ FLOAT outData[1]; // 每个block只保留累加的结果，即一个值

    /**block 是计算的基础单元 <<<2,256>>>*/
    // int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // if(bid>=height*width) return; // 超出边界

    // 根据bid 反算出image的行与列
    int row=bid/width;
    int col=bid%width;

    FLOAT img_value=0;
    int cur_row=0;
    int cur_col=0;

    // 全局内存--->共享内存
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
            
            imgData[j+i*kernel_size]=img_value;
            wData[j+i*kernel_size]=dev_kernel[j+i*kernel_size];
        }
    }
    
    /*
    if (tid==0)
    {
        for(int i=0;i<SIZE;++i)
        {
            // atomicAdd(&outData[0],imgData[i]*wData[i]);
            outData[0]+=imgData[i]*wData[i]; // 一个线程做，不会有冲突
        }
    }
   //*/ 

    atomicAdd(&outData[0],imgData[tid]*wData[tid]); // block内每个线程同时做，在将结果相加，atomicAdd避免线程冲突

    if (tid==0)
        // 共享内存到全局内存
        dev_z[bid] = outData[0];
}


// 卷积转成矩阵乘法 每次处理一块 大小 32x9
template <int SIZE1,int SIZE2>
__global__ void shared_conv2(FLOAT *dev_x,FLOAT *dev_z,FLOAT *dev_kernel,int height,int width,int kernel_size)
{
    __shared__ FLOAT imgData[SIZE1][SIZE2]; // 32x9
    __shared__ FLOAT wData[SIZE1][SIZE2];
    __shared__ FLOAT outData[SIZE1];

    // if(bid>=height*width) return; // 超出边界
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int bid = bx + by*gridDim.x;

    // 根据bid 反算出image的行与列
    int row=bid/width;
    int col=bid%width;

    FLOAT img_value=0;
    int cur_row=0;
    int cur_col=0;

    // 全局内存--->共享内存
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
            
            imgData[ty][j+i*kernel_size]=img_value;
            wData[ty][j+i*kernel_size]=dev_kernel[j+i*kernel_size];
        }
    }
    
    FLOAT csum=0;
    for(int i=0;i<SIZE2;++i) // 单线程
        outData[ty]+=imgData[ty][i]*wData[ty][i]; // 同一个线程中执行 不会有冲突
        // or
        // csum+=imgData[ty][i]*wData[ty][i];
    

    /*
    // 多个线程同时执行会有冲突
    // csum+=imgData[ty][tx]*wData[ty][tx];
    // outData[ty]+=imgData[ty][tx]*wData[ty][tx];
    // 使用原子加解决冲突
    atomicAdd(&outData[ty],imgData[ty][tx]*wData[ty][tx]); //使用了原子变量速度反而更慢
    */
    // 共享内存到全局内存
    dev_z[bid] = outData[ty];
    // dev_z[bid] = csum;
}