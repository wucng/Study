#include "function.h"

/* 直接放到了../../../common/cudaFunction.cuh定义(同时做了声明与定义)
// Add two vectors on the GPU
template <typename T>
__global__ void vectorAddGPU(T *a, T *b, T *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // if (idx>=N) return;

    // if (idx < N)
    // {
    //     c[idx] = a[idx] + b[idx];
    // }
    while(idx<N)
    {
        c[idx] = a[idx] + b[idx];
        idx+=blockDim.x*gridDim.x;
    }
}
*/


template <typename T>
void exec_vector_add(Data<T>& data)
{   
    // 加上测时间
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((data.nums+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    vectorAddGPU<T><<<grid,block>>>(data.h_a,data.h_b,data.h_c,data.nums);
    checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    // checkCudaErrors(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}


int main(int argc,char *argv[])
{   
    int nums=5,flag=0;
    if(argc>2)
    {
        nums=atoi(argv[1]);
        flag=atoi(argv[2]);
    }
    Data<FLOAT> data;
    data.allocate(nums,flag);
    exec_vector_add<FLOAT>(data);
    //print
    data.pprint();

    return 0;
}
