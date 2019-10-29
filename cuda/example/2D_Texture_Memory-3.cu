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
/**声明2维纹理内存*/
texture<FLOAT,2>  tex2DRef;


__global__ void kernel(FLOAT *dev_b,const int height,const int width)
{
    int x=threadIdx.x;
    int y=threadIdx.y;
    int idx=x+y*blockDim.x;
    if(idx>=N) return;
    // if(x>=width || y>= height) return;
    dev_b[idx]=tex2D(tex2DRef,x,y)+2;
}

int main()
{
	/**一维矩阵按二维格式存储成纹理内存，其实二维数组也是按一维数组格式存放*/
    int width=5;
    int height=N/width;
    FLOAT *host_a=NULL,*dev_a=NULL;
    int nBytes=N*sizeof(FLOAT);
    HANDLE_ERROR(cudaMallocHost((void **)&host_a,nBytes));
    for(int i=0;i<N;++i) host_a[i]=1;

    // 将host 拷贝到GPU全局内存中
    HANDLE_ERROR(cudaMalloc((void **)&dev_a,nBytes));

    // 纹理内存绑定全局内存
    // cudaChannelFormatDesc desc = cudaCreateChannelDesc<FLOAT>();
    size_t pitch;
    HANDLE_ERROR( cudaMallocPitch((void**)&dev_a, &pitch, width*sizeof(FLOAT), height) );
    HANDLE_ERROR( cudaMemcpy2D(dev_a,             // device destination
                             pitch,           // device pitch (calculated above)
                             host_a,               // src on host
                             width*sizeof(float), // pitch on src (no padding so just width of row)
                             width*sizeof(float), // width of data in bytes
                             height,            // height of data
                             cudaMemcpyHostToDevice) );
    HANDLE_ERROR(cudaBindTexture2D(NULL,tex2DRef,dev_a,tex2DRef.channelDesc,
                                   width,height,pitch));

    FLOAT *dev_b=NULL,*host_b=NULL;
    HANDLE_ERROR(cudaMallocHost((void **)&host_b,nBytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b,nBytes));

    dim3 threads(32,32);
    kernel<<<1,threads>>>(dev_b,N/width,width);
    HANDLE_ERROR(cudaMemcpy(host_b,dev_b,nBytes,cudaMemcpyDeviceToHost));
	// cudaDeviceSynchronize(); //等待GPU执行完成， 有多种方式

    // print
    for(int i=0;i<N;++i) cout<<host_b[i]<<" ";

    HANDLE_ERROR(cudaUnbindTexture(tex2DRef));// 解除绑定
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    return 0;
}
