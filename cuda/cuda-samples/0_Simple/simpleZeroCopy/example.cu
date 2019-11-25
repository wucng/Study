/**
* 这个例子说明了如何使用 零内存拷贝 ，kernels可以直接读写固定的系统内存
* 1.固定的系统分页内存 （多种CPU内存分配方式）
* 2.nvcc example.cu -I ../../common/ -std=c++11
* 3. ./a.out 10 1
*/

#include "common.h"
#include "cudaFunction.cuh"

typedef float FLOAT;


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
