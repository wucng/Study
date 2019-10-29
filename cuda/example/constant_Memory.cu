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

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

using namespace std;
typedef float FLOAT;
#define N 20
__constant__ FLOAT A[N]; // 声明常量内存

__global__ void kernel(FLOAT *dev_b)
{
    int idx=threadIdx.x;
    if(idx>=N) return;
    dev_b[idx]=A[idx]+2;
}

int main()
{
    FLOAT *host_a=NULL;
    int nBytes=N*sizeof(FLOAT);
    HANDLE_ERROR(cudaMallocHost((void **)&host_a,nBytes));
    for(int i=0;i<N;++i) host_a[i]=1;

    // 将host 拷贝到GPU
    // cudaMemcpy()  ,cudaMemcpyHostToDevice默认拷贝到GPU的全局内存
    HANDLE_ERROR(cudaMemcpyToSymbol(A,host_a,nBytes));// host 到GPU常量内存

    FLOAT *dev_b=NULL,*host_b=NULL;
    HANDLE_ERROR(cudaMallocHost((void **)&host_b,nBytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b,nBytes));

    kernel<<<1,32>>>(dev_b);
    HANDLE_ERROR(cudaMemcpy(host_b,dev_b,nBytes,cudaMemcpyDeviceToHost));
	// cudaDeviceSynchronize(); //等待GPU执行完成， 有多种方式

    // print
    for(int i=0;i<N;++i) cout<<host_b[i]<<" ";

    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    return 0;
}
