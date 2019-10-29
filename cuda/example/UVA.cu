#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>
// #include "common/book.h"

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

/* 全局线程id get thread id: 1D block and 2D grid  <<<(32,32),32>>>*/
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)  // 2D grid,1D block
// #define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x+threadIdx.y*blockDim.x)  // 2D grid,2D block

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* 每个block的线程id*/
// #define get_tid_per_block() (threadIdx.x+threadIdx.y*blockDim.x) // 2D block
#define get_tid_per_block() (threadIdx.x)

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

using namespace std;
typedef float FLOAT;


__global__ void vec_add(FLOAT *A,const int N)
{
    int idx=get_tid();
    if(idx>=N) return;
    A[idx]+=2;
}

int main()
{
    mycout<<"虚拟统一内存使用(CPU与GPU都能访问)\n"<<
    "1.使用cudaMallocManaged分配内存\n"<<
    "2.第3个参数cudaMemAttachGlobal or cudaMemAttachHost\n"<<
    "3.(defaults to cudaMemAttachGlobal)"<<endl;

     int N=10;
    int nBytes=N*sizeof(FLOAT);

    FLOAT *host_and_dev_data=NULL;

    // GPU分配内存空间
    CHECK(cudaMallocManaged((void**)&host_and_dev_data,nBytes));

    // 赋值
    for(int i=0;i<N;++i)
    {
        host_and_dev_data[i]=1;
    }

    // 启动核函数
    vec_add<<<1,32>>>(host_and_dev_data,N);


    cudaDeviceSynchronize(); //等待GPU执行完成， 有多种方式

    // printf
    for(int i=0;i<N;++i)
    {
        cout<<host_and_dev_data[i]<<" ";
    }
    cout<<endl;

    CHECK(cudaFree(host_and_dev_data));

    return 0;
}
