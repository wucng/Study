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

__global__ void global_atmoic_add(FLOAT *dev_x,FLOAT *dev_z,int N)
{
    /**block 是计算的基础单元*/
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    //int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    //int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    // 原子操作，会自动解决同步问题（可以实现block内部线程同步，以及block之间线程同步）
    if (idx<N)
        atomicAdd(&dev_z[0],dev_x[idx]); // dev_z[0]+=dev_x[idx]
}


__global__ void shared_atmoic_add(FLOAT *dev_x,FLOAT *dev_z,int N)
{
    extern __shared__ FLOAT sdatas[];// 声明共享内存（每个block 都有自己独立的共享内存）

    /**block 是计算的基础单元*/
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    // int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    /*
    {
        // 每个block将对应的全局变量复制到每个block对应的共享内存（每个block 都有自己独立的共享内存）
        sdatas[tid]=idx<N?dev_x[idx] : 0;
        __syncthreads();

        // 原子操作，会自动解决同步问题（可以实现block内部线程同步，以及block之间线程同步）
        if (tid<blockDim.x)
            atomicAdd(&dev_z[0],sdatas[tid]);
    }
    */

    {
        // 每个block将对应的全局变量复制到每个block对应的共享内存（每个block 都有自己独立的共享内存）
        sdatas[tid]=idx<N?dev_x[idx]+dev_x[idx+blockDim.x/2] : 0; // 写入共享内存时把数据长度缩减一半可以加快速度
        __syncthreads();

        // 原子操作，会自动解决同步问题（可以实现block内部线程同步，以及block之间线程同步）
        if (tid<blockDim.x/2)
            atomicAdd(&dev_z[0],sdatas[tid]);
    }
}

int main()
{
    mycout<<"原子操作计算数组的和"<<endl;
    const int N=200000; // 向量长度
    int nbytes=N*sizeof(FLOAT);
    int onebytes=1*sizeof(FLOAT);

    FLOAT *host_x=NULL,*host_z=NULL;
    FLOAT *dev_x=NULL,*dev_z=NULL;

    /**1D block*/
    int bs=1024;

    /**1D grid*/
    int grid=ceil(1.0*N/bs);

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, onebytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, onebytes));

    for(int i=0;i<N;++i)
    {
        host_x[i]=1;
    }

    // CPU-->GPU
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成

    // CPU 启动 GPU kernel计算
    {
        // 使用全局变量
        // global_atmoic_add<<<grid,bs>>>(dev_x,dev_z,N);

        // 使用共享内存变量
        shared_atmoic_add<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_z,N);
    }

    // GPU ---> CPU
    HANDLE_ERROR(cudaMemcpy(host_z, dev_z, onebytes, cudaMemcpyDeviceToHost));

    // 打印结果
    cout << host_z[0]<<endl;

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}