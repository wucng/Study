#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>
#include "common/book.h"

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "

using namespace std;
typedef float FLOAT;
// #define N 20
/**声明2维纹理内存*/
texture<FLOAT,2>  tex2DRef;

__global__ void kernel(FLOAT *dev_b,const int height,const int width)
{
    int x=threadIdx.x;
    int y=threadIdx.y;
    int idx=x+y*blockDim.x;
    if(idx>=width*height) return;
    // if(x>=width || y>= height) return;
    dev_b[idx]=tex2D(tex2DRef,x,y)+2;
}

int main()
{
    const int N=20;
    int width=5;
    int height=N/width;
    int nBytes=N*sizeof(FLOAT);
    FLOAT (*host_a)[width];
    // FLOAT (*d_f)[width],
    FLOAT *dev_a;
    HANDLE_ERROR(cudaMallocHost((void **)&host_a,nBytes));

    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            host_a[i][j]=1;
        }
    }

    // 纹理内存绑定全局内存
    size_t pitch;
    HANDLE_ERROR( cudaMallocPitch((void**)&dev_a, &pitch, width*sizeof(FLOAT), height) );
    HANDLE_ERROR( cudaMemcpy2D(dev_a,             // device destination
                             pitch,           // device pitch (calculated above)
                             host_a,               // src on host
                             width*sizeof(float), // pitch on src (no padding so just width of row)
                             width*sizeof(float), // width of data in bytes
                             height,            // height of data
                             cudaMemcpyHostToDevice) );

    // cudaChannelFormatDesc desc = cudaCreateChannelDesc<FLOAT>();
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
