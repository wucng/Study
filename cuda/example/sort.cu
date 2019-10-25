#include <iostream>
#include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
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


__global__
void global_sort(const FLOAT *dev_x,FLOAT *dev_z,const int N)
{
    // 由于存在跨block访问(block>1)，因此不能使用共享内存方式
    // 如果block=1，则不存在跨block访问，可以使用共享内存
    // 如果要使用共享内存，则必须消除跨block访问的问题，否则不能使用共享内存方式
    int idx=get_tid();
    if(idx>=N) return; // 越界问题处理

    dev_z[idx]=dev_x[idx];
    __syncthreads(); // 同步

    FLOAT tmp=0; // 每个线程内的局部变量
    int i=0;
    for(i=0;i<N;++i)
    {
        if(idx+1>=N) continue; // 越界处理

        if(i%2==idx%2)
        {
            if (dev_z[idx]>=dev_z[idx+1])
            {
                // 交换
                tmp=dev_z[idx];
                dev_z[idx]=dev_z[idx+1];
                dev_z[idx+1]=tmp;
            }
            __syncthreads(); // 同步
        }
    }
}

/**或者这样*/
__global__ void global_sort2(const FLOAT *dev_x,FLOAT *dev_z,const int N)
{
    // 由于存在跨block访问(block>1)，因此不能使用共享内存方式
    // 如果block=1，则不存在跨block访问，可以使用共享内存
    // 如果要使用共享内存，则必须消除跨block访问的问题，否则不能使用共享内存方式
    int idx=get_tid();
    if(idx>=N) return; // 越界问题处理

    dev_z[idx]=dev_x[idx];
    __syncthreads(); // 同步

    // 保证tmp1<tmp2（从小到大排序）
    FLOAT tmp1=0; // 每个线程内的局部变量
    FLOAT tmp2=0; // 每个线程内的局部变量
    int i=0;
    for(i=0;i<N;++i)
    {
        if(idx+1>=N) continue; // 越界处理 (使用return会导致线程退出)

        // 先不写入，等待这一组完成后在统一写入，避免读写冲突问题
        if (dev_z[idx]>=dev_z[idx+1])
        {
            tmp1=dev_z[idx+1];
            tmp2=dev_z[idx];
        }
        else
        {
            tmp1=dev_z[idx];
            tmp2=dev_z[idx+1];
        }
        __syncthreads(); // 同步

        // 写入
        if(i%2==idx%2)
        {
            dev_z[idx]=tmp1;
            dev_z[idx+1]=tmp2;
        }
        __syncthreads(); // 同步
    }
}

__global__
void shared_sort(const FLOAT *dev_x,FLOAT *dev_z,const int N)
{
    // 如果block>1,会存在跨block访问的情况，而无法使用共享内存
    // 如果block=1,不存在跨block访问的情况，可以使用共享内存

    extern __shared__ FLOAT sdatas[];// 声明共享内存
    // 由于存在跨block访问(block>1)，因此不能使用共享内存方式
    // 如果block=1，则不存在跨block访问，可以使用共享内存
    // 如果要使用共享内存，则必须消除跨block访问的问题，否则不能使用共享内存方式
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    //int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    if(idx>=N) return; // 越界问题处理

    sdatas[tid]=dev_x[idx];
    __syncthreads(); // 同步

    FLOAT tmp=0; // 每个线程内的局部变量
    int i=0;
    for(i=0;i<N;++i)
    {
        if(tid+1>=N) continue; // 越界处理

        if(i%2==tid%2)
        {
            if (sdatas[tid]>=sdatas[tid+1])
            {
                // 交换
                tmp=sdatas[tid];
                sdatas[tid]=sdatas[tid+1];
                sdatas[tid+1]=tmp;
            }
            __syncthreads(); // 同步
        }
    }

    // 再写入到全局变量
    dev_z[idx]=sdatas[tid];
}

int main()
{
    mycout<<"scan实现"<<endl;

    int N=10;

    FLOAT *host_x=NULL,*host_z=NULL;
    FLOAT *dev_x=NULL,*dev_z=NULL;

    /**1D block*/
    int bs=1024;

    /**1D grid*/
    int grid=ceil(1.0*N/bs);

    int nbytes=N*sizeof(FLOAT);

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, nbytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, nbytes));

    // 随机种子
    srand((unsigned int )time(NULL));
    // 赋值
    for(int i=0;i<N;++i)
    {
        host_x[i]=rand()%256; // [0,256)
    }

    // CPU-->GPU
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成


    // CPU 启动 GPU kernel计算
    {
        if (grid>1)
        {
            // 使用全局变量
            mycout<<"grid = "<<grid<<"\n使用全局内存"<<endl;
            global_sort<<<grid,bs>>>(dev_x,dev_z,N);
            // global_sort2<<<grid,bs>>>(dev_x,dev_z,N);
        }
        else
        {
            // 如果blockDim.x>1 会存在跨block访问情况，共享内存没法实现block间通信(不能使用共享内存，只能使用全局内存)
            // 如果blockDim.x=1，也就是只有一个block 不存在跨block访问情况,可以使用共享内存
            mycout<<"grid = "<<grid<<"\n使用共享内存"<<endl;
            // 使用共享内存变量 (共享内存变量只能block内部通信，跨block没法访问)
            shared_sort<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_z,N);
        }
    }

    // GPU-->CPU
    HANDLE_ERROR(cudaMemcpy(host_z,dev_z, nbytes, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaDeviceSynchronize());


    // 打印部分结果
    mycout<<"before sort"<<endl;
    for(int i=0;i<N;++i)
    {
        cout<<host_x[i]<<" ";
    }
    cout <<endl;

    mycout<<"after sort"<<endl;
    for(int i=0;i<N;++i)
    {
        cout<<host_z[i]<<" ";
    }
    cout <<endl;

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}
