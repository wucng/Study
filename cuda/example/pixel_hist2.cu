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
typedef int FLOAT;

__global__
void global_hist(const FLOAT *dev_x,FLOAT *dev_z,const int N)
{
    /**block 是计算的基础单元*/
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    //int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    //int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    if(idx>=N) return; // 超出边界

    // dev_z[dev_x[idx]]+=1;
    // 原子操作，会自动解决同步问题（可以实现block内部线程同步，以及block之间线程同步）
    atomicAdd(&dev_z[dev_x[idx]],1);
}

__global__
void global_vec_add(const FLOAT *dev_z1,const FLOAT *dev_z2,FLOAT *dev_z,const int hist_bin)
{
     int idx = get_tid();
     if(idx>=hist_bin) return; // 超出边界

     dev_z[idx]=dev_z1[idx]+dev_z2[idx];
}

int main()
{
    mycout<<"实现像素直方图统计"<<endl;
    mycout << "说明：\n"<<
                "0、当直方图分的bin数太少时，可以先计算局部直方图，再计算全局直方图，可以增加计算的并行性，提高计算效率\n"<<
                 "1、将一个大的数组均分成几份，每份启动一个kernel计算得到局部直方图"<<
                " (采用异步流同时处理，同步流 必须等待上一个kernel计算完成 才会计算下一个kernel)\n"<<
                 "2、再将每部分的局部直方图对应相加得到最终的直方图\n"<<endl;

	const int height=1080; // image 高
    const int width=1920; // image 宽
    // const int height=20; // image 高
    // const int width=20; // image 宽

    int hist_bin=256; // 0~256 统计每个像素的数量，得到频率直方图

    FLOAT *host_x=NULL,*host_z=NULL;
    FLOAT *dev_x1=NULL,*dev_z1=NULL; // 第一部分
    FLOAT *dev_x2=NULL,*dev_z2=NULL; // 第二部分
    FLOAT *dev_z=NULL; // 合并最终结果

    // 这里只分成2个部分计算局部直方图，也可以分成更多个部分
    int N=height*width;
    int N1=N/2;
    int N2=N-N/2;

    /**1D block*/
    int bs=1024;

    /**1D grid*/
    int grid1=ceil(1.0*N1/bs);
    int grid2=ceil(1.0*N2/bs);

    // 把二维展开成一维 （数组内存都是按一维排列）
    int nbytes=N*sizeof(FLOAT);
    int n1bytes=N1*sizeof(FLOAT);
    int n2bytes=N2*sizeof(FLOAT);
    int binbytes=hist_bin*sizeof(FLOAT);

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, binbytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x1, n1bytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_x2, n2bytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z1, binbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z2, binbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, binbytes));

    // 设置随机种子
    srand((unsigned int) time(NULL) );

    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            host_x[j+i*width]= rand()%hist_bin; // [0,255) 随机数
        }
    }

    // CPU-->GPU（把数据分成2份）
    // 默认是启动同步流(只有1个流)，我们要使用异步流处理(不需要同步，每个kernel计算各自的局部直方图)
    // HANDLE_ERROR(cudaMemcpy(dev_x1, host_x, n1bytes, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(dev_x2, host_x+N1, n2bytes, cudaMemcpyHostToDevice));// 指针开始位置是 host_x+N1

    // 启动多个流
    cudaStream_t s[2];
    HANDLE_ERROR(cudaStreamCreate(&s[0]));
    HANDLE_ERROR(cudaStreamCreate(&s[1]));

    HANDLE_ERROR(cudaMemcpyAsync(dev_x1, host_x, n1bytes, cudaMemcpyHostToDevice,s[0]));
    HANDLE_ERROR(cudaMemcpyAsync(dev_x2, host_x+N1, n2bytes, cudaMemcpyHostToDevice,s[1]));

    HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成


    // CPU 启动 GPU kernel计算
    {
        // 使用全局变量
        global_hist<<<grid1,bs,0,s[0]>>>(dev_x1,dev_z1,N1); // 第3个参数为分配的共享内存大小，第4个参数指定使用哪个流
        global_hist<<<grid2,bs,0,s[1]>>>(dev_x2,dev_z2,N2); // 如果要指定流，第3个参数不能省略(否则会把流ID当初共享内存大小而报错)

        // 使用共享内存变量 (共享内存变量只能block内部通信，跨block没法访问)
        // shared_hist<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_z,height,width);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());

    {
        // 将局部直方图对应相加(向量的加法)得到最终的直方图
        // global_vec_add<<<1,bs>>>(dev_z1,dev_z2,dev_z,hist_bin);
        // GPU ---> CPU
        // HANDLE_ERROR(cudaMemcpy(host_z, dev_z, binbytes, cudaMemcpyDeviceToHost));

        // or
        global_vec_add<<<1,bs,0,s[0]>>>(dev_z1,dev_z2,dev_z,hist_bin);
        // GPU ---> CPU
        HANDLE_ERROR(cudaMemcpyAsync(host_z, dev_z, binbytes, cudaMemcpyDeviceToHost,s[0]));
    }

    // 打印部分结果
    mycout<<"result"<<endl;
    int sum=0;
    for(int i=0;i<hist_bin;++i)
    {
        sum+=host_z[i];
    }
    cout <<endl;
    cout <<sum<<endl;

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x1));
    HANDLE_ERROR(cudaFree(dev_x2));
    HANDLE_ERROR(cudaFree(dev_z1));
    HANDLE_ERROR(cudaFree(dev_z2));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    // 销毁流
    HANDLE_ERROR(cudaStreamDestroy(s[0]));
    HANDLE_ERROR(cudaStreamDestroy(s[1]));

    return 0;
}
