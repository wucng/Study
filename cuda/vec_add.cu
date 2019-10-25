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

/**全局内存*/
__global__ void global_vec_add(FLOAT *x,FLOAT *y,FLOAT *z,int N)
{
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    // 以block为计算单元，每个block的线程数为 blockDim.x
    // if(idx<N)
    //    z[idx]=x[idx]+y[idx];
    //// __syncthreads(); // 不需要读取其他线程上的数据，因此不需同步操作

    // 如果不能一次计算完，必须使用循环串行运行，每次执行 gridDim.x*blockDim.x 个任务
    while(idx<N)
    {
        z[idx]=x[idx]+y[idx];
        idx+=gridDim.x*blockDim.x;
    }


    // 不需要同步 因此不使用 __syncthreads(); // block 内部线程同步
}


/**共享内存*/
__global__ void shared_vec_add(FLOAT *x,FLOAT *y,FLOAT *z,int N)
{
    extern __shared__ FLOAT sdatas[];// 声明共享内存

    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    /*
    // 每个block将对应的全局变量复制到共享内存
    sdatas[tid]=idx<N?x[idx]+y[idx] : 0;
	__syncthreads();
	
    // 再将各block的共享内存变量拷贝给全局内存变量
    if(idx<N)
        z[idx]=sdatas[tid];
    */

    // 如果不能一次计算完，必须使用循环串行运行，每次执行 gridDim.x*blockDim.x 个任务
    while(idx<N)
    {	// 如果会造成线程冲突，必须使用同步操作，否则不要使用同步(同步会降低效率)
        // 每个block将对应的全局变量复制到共享内存
        sdatas[tid]=idx<N?x[idx]+y[idx] : 0;
        __syncthreads();

        // 再将各block的共享内存变量拷贝给全局内存变量
        if(idx<N)
            z[idx]=sdatas[tid];
        __syncthreads(); // 必须同步，对于共享内存，上一步的共享内存必须取完后，在把新的全局变量写入
		// 访问全局变量不会有冲突，但访问共享内存会有冲突 必须同步
        idx+=gridDim.x*blockDim.x;
    }
}



int main()
{
    mycout<<"向量加法"<<endl;
    const int N=1000; // 向量长度
    int nbytes=N*sizeof(FLOAT);

    FLOAT *host_x=NULL,*host_y=NULL,*host_z=NULL;
    FLOAT *dev_x=NULL,*dev_y=NULL,*dev_z=NULL;

    /**1D block*/
    // int bs=1024;
    int bs=32;

    /**1D grid*/
    // int grid=ceil(1.0*N/bs);
    int grid=1;

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_y, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, nbytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_y, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, nbytes));

    for(int i=0;i<N;++i)
    {
        host_x[i]=2;
        host_y[i]=3;
    }

    // CPU-->GPU
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_y, host_y, nbytes, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成

    // CPU 启动 GPU kernel计算
    {
        // 使用全局变量
        // global_vec_add<<<grid,bs>>>(dev_x,dev_y,dev_z,N);

        // 使用共享内存变量
        shared_vec_add<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_y,dev_z,N);
    }

    // GPU ---> CPU
    HANDLE_ERROR(cudaMemcpy(host_z, dev_z, nbytes, cudaMemcpyDeviceToHost));

    // 打印结果
    for(int i=0;i<10;++i)
    {
        cout << host_x[i] << " + " <<host_y[i]<<" = "<<host_z[i]<<endl;
    }
    cout<<endl;

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_y));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_y));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}



