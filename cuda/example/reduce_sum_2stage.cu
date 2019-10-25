#include <iostream>
// #include <iomanip>
#include <stdio.h>
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

/**使用全局内存*/
__global__ void global_reduce_sum(FLOAT *d_in,FLOAT *d_out,int N)
{
    // d_in,d_out 都是(GPU内)全局变量
    /**
    * 1、全局内存使用全局索引访问，主要用于block之间通信
    *     （block内部线程间也是可以通信，但效率低，block内部通信完全可以使用共享内存方式）
    * 2、共享内存主要用于block内部各线程之间的通信，针对每个block内部
    *         而block内部线程总数为 blockDim.x*blockDim.y (1维block，则blockDim.y=1)
    *         block内部线程同步使用 __syncthreads()
    * 3、1个block内每32个线程组成1个warp，执行相同的指令，每个warp内是自动同步
    *         因此不需使用 __syncthreads() 减少开销提高效率
    * 4、如果是内存对齐，不需要跨block访问数据，使用全局内存，以每个线程为单元，如：向量加法
    *        如果是要跨block访问数据，使用共享内存(block内部线程同步)，以block为单元，如：归约
    */
	// 定义每个线程的局部变量（每个线程的局部内存）
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    /**
    * 假设<<<128,256>>>
    * 1、先计算每个block内所有线程的结果,计算完成后会有128个中间结果
    * 2、如果1计算的中间结果比较大，可以重复1操作
    * 3、在将这128个值放在1个block(256个线程)计算得到最后的结果
    */

    // 以每个block为计算单元
    for(unsigned int s=blockDim.x/2;s>0;s>>=1) // s/=2
    {
        if(tid<s & idx + s<N)
		// if(tid<s)
            d_in[idx]+=d_in[idx + s];
        __syncthreads(); // block 内部线程同步
    }

    // 每个block 取第0个线程赋值，获取中间结果
    if(tid==0)
        d_out[bid]=d_in[idx];

}

/**block内使用共享内存*/
__global__ void share_reduce_sum(FLOAT *d_in,FLOAT *d_out,int N)
{
    // extern __shared__ FLOAT sdatas[]; // 声明共享内存
    extern __shared__ volatile FLOAT sdatas[]; // 使用了循环展开需加上volatile 字段，否则结果不正确

    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    // 每个block将对应的全局内存复制到共享内存
    sdatas[tid]=idx<N?d_in[idx] : 0;
    __syncthreads();

    /**
    * 假设<<<128,256>>>
    * 1、先计算每个block内所有线程的结果,计算完成后会有128个中间结果
    * 2、如果1计算的中间结果比较大，可以重复1操作
    * 3、在将这128个值放在1个block(256个线程)计算得到最后的结果
    */

    /*
    // 以每个block为计算单元
    for(unsigned int s=blockDim.x/2;s>0;s>>=1) // s/=2
    {
        if(tid<s & idx + s<N)
		// if(tid<s)
            sdatas[tid]+=sdatas[tid + s];
        __syncthreads(); // block 内部线程同步
    }
    */

    // 以每个block为计算单元
    for(unsigned int s=blockDim.x/2;s>32;s>>=1) // s/=2
    {
        if(tid<s & idx + s<N)
		// if(tid<s)
            sdatas[tid]+=sdatas[tid + s];
        __syncthreads(); // block 内部线程同步
    }

    // 32个线程对应1个warp 可以直接展开，且会自动同步，不需要使用__syncthreads()
    if (tid<32)
    {
        sdatas[tid]+=sdatas[tid + 32];
        sdatas[tid]+=sdatas[tid + 16];
        sdatas[tid]+=sdatas[tid + 8];
        sdatas[tid]+=sdatas[tid + 4];
        sdatas[tid]+=sdatas[tid + 2];
        sdatas[tid]+=sdatas[tid + 1];
    }

    // 每个block 取第0个线程赋值，获取中间结果
    if(tid==0)
        d_out[bid]=sdatas[0];

}


/**block内使用共享内存 完全展开*/
__global__ void share_reduce_sum2(FLOAT *d_in,FLOAT *d_out,int N)
{
    extern __shared__ volatile FLOAT sdatas[]; // 使用了循环展开需加上volatile 字段，否则结果不正确

    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    /**说明
    * idx，tid，bid 每个线程的局部变量(线程的局部内存) -- 针对每个线程
    * d_in，d_out 全局变量（全局内存）-- 针对所有线程（包括跨block之间的线程）
    * sdatas 每个block内线程的共享变量（共享内存）-- 针对每个block内部的所有线程
    */

    // 每个block将对应的全局内存复制到共享内存
    sdatas[tid]=idx<N?d_in[idx] : 0;
    __syncthreads();

    /*
    // 以每个block为计算单元
    for(unsigned int s=blockDim.x/2;s>32;s>>=1) // s/=2
    {
        if(tid<s & idx + s<N)
		// if(tid<s)
            sdatas[tid]+=sdatas[tid + s];
        __syncthreads(); // block 内部线程同步
    }
    */

    // blockDim.x=1024
    if(tid<512)
        sdatas[tid]+=sdatas[tid + 512];
    __syncthreads();

    if(tid<256)
        sdatas[tid]+=sdatas[tid + 256];
    __syncthreads();

    if(tid<128)
        sdatas[tid]+=sdatas[tid + 128];
    __syncthreads();

    if(tid<64)
        sdatas[tid]+=sdatas[tid + 64];
    __syncthreads();

    // 32个线程对应1个warp 可以直接展开，且会自动同步，不需要使用__syncthreads()
    if (tid<32)
    {
        sdatas[tid]+=sdatas[tid + 32];
        sdatas[tid]+=sdatas[tid + 16];
        sdatas[tid]+=sdatas[tid + 8];
        sdatas[tid]+=sdatas[tid + 4];
        sdatas[tid]+=sdatas[tid + 2];
        sdatas[tid]+=sdatas[tid + 1];
    }

    // 每个block 取第0个线程赋值，获取中间结果
    if(tid==0)
        d_out[bid]=sdatas[0];

}


/**block内使用共享内存 完全展开*/
__global__ void share_reduce_sum3(FLOAT *d_in,FLOAT *d_out,int N)
{
    extern __shared__ volatile FLOAT sdatas[]; // 使用了循环展开需加上volatile 字段，否则结果不正确

    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    /**说明
    * idx，tid，bid 每个线程的局部变量(线程的局部内存) -- 针对每个线程
    * d_in，d_out 全局变量（全局内存）-- 针对所有线程（包括跨block之间的线程）
    * sdatas 每个block内线程的共享变量（共享内存）-- 针对每个block内部的所有线程
    */

    // 每个block将对应的全局内存复制到共享内存
    // 在写入共享内存时就做一次加法操作，减少数据量，加快速度
    sdatas[tid]=idx<N?d_in[idx]+d_in[idx + blockDim.x/2] : 0;
    __syncthreads();

    /*
    // 以每个block为计算单元
    for(unsigned int s=blockDim.x/2;s>32;s>>=1) // s/=2
    {
        if(tid<s & idx + s<N)
		// if(tid<s)
            sdatas[tid]+=sdatas[tid + s];
        __syncthreads(); // block 内部线程同步
    }
    */

    // blockDim.x=1024
    /** 在写入共享内存时 已经做过了这一步操作
    if(tid<512)
        sdatas[tid]+=sdatas[tid + 512];
    __syncthreads();
    */
    if(tid<256)
        sdatas[tid]+=sdatas[tid + 256];
    __syncthreads();

    if(tid<128)
        sdatas[tid]+=sdatas[tid + 128];
    __syncthreads();

    if(tid<64)
        sdatas[tid]+=sdatas[tid + 64];
    __syncthreads();

    // 32个线程对应1个warp 可以直接展开，且会自动同步，不需要使用__syncthreads()
    if (tid<32)
    {
        sdatas[tid]+=sdatas[tid + 32];
        sdatas[tid]+=sdatas[tid + 16];
        sdatas[tid]+=sdatas[tid + 8];
        sdatas[tid]+=sdatas[tid + 4];
        sdatas[tid]+=sdatas[tid + 2];
        sdatas[tid]+=sdatas[tid + 1];
    }

    // 每个block 取第0个线程赋值，获取中间结果
    if(tid==0)
        d_out[bid]=sdatas[0];

}


int main(int argc, char *argv[])
{
    mycout<<"归约算法求和 2阶段求和"<<endl;
    int N=980000; // 使用二阶段 最多取值为 1024*1024
    int nbytes = N * sizeof(FLOAT);

    // 阶段一
    /* 1D block 每个block是256x1个thread*/
    int bs = 1024; // 必须是(32*2^n)，最大1024(需查看显卡)
    int num_bs=bs;

    /* 1D grid */
    int grid=ceil(1.0*N / num_bs);
    int num_grid=grid;

    /* 2D grid*/
    // int s=ceil(sqrt(1.0*N / num_bs));
    // dim3 grid(s,s);
    // int num_grid=s*s;

    // 阶段二汇总
    int grid2=1;

    int gbytes = num_grid * sizeof(FLOAT); // 中间结果
    int onebytes = 1 * sizeof(FLOAT); // 最终结果

    /**======1、CPU 创建变量并赋值初始化==========*/
    FLOAT *dev_x=NULL,*host_x=NULL;
    FLOAT *dev_y=NULL; // 阶段一临时中间变量
    FLOAT  *dev_z=NULL,*host_z=NULL;// 最终结果

    // 普通malloc分配内存速度比 cudaMallocHost慢
    // host_x=(FLOAT*)malloc(nbytes);
    // host_z=(FLOAT*)malloc(onebytes);
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, onebytes));

    /*给CPU变量赋值*/
    for (int i=0; i<N; i++) {
        host_x[i] = 1;
    }

    /**======2、GPU 变量分配内存======*/
    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_y, gbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, onebytes));

    /**强制小数格式显示，并保留2个小数精度*/
    cout << fixed ;// << setprecision(2);
    cout.precision(3);
    // printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
    // cout<<"allocated "<< (nbytes / (1024.f * 1024.f))<<" MB on GPU"<<endl;

    /**======3、将CPU数据拷贝给GPU======*/
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));

    /**======4、调用GPU计算======*/
    HANDLE_ERROR(cudaDeviceSynchronize()); // 等待CPU数据完全拷贝给GPU
    clock_t start = clock();
    {
        // stage 1
        // global_reduce_sum<<<grid,bs>>>(dev_x,dev_y,N);
        // share_reduce_sum<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_y,N);
        // share_reduce_sum2<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_y,N);
        share_reduce_sum3<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_y,N);
    }
    {
        // stage 2
        // global_reduce_sum<<<grid2,bs>>>(dev_y,dev_z,num_grid);
        // share_reduce_sum<<<grid2,bs,bs*sizeof(FLOAT)>>>(dev_y,dev_z,num_grid);
        // share_reduce_sum2<<<grid2,bs,bs*sizeof(FLOAT)>>>(dev_y,dev_z,num_grid);
        share_reduce_sum3<<<grid2,bs,bs*sizeof(FLOAT)>>>(dev_y,dev_z,num_grid);
    }

    // HANDLE_ERROR(cudaDeviceSynchronize()); // CPU等待GPU计算完成

    /**======5、GPU计算结果拷贝给CPU======*/
    HANDLE_ERROR(cudaMemcpy(host_z,dev_z, onebytes, cudaMemcpyDeviceToHost));

    // cout.precision(15);
    mycout<<"GPU cost time:"<<(double)(clock()-start)/CLOCKS_PER_SEC <<"s"<<endl;

    // 打印结果
    cout << host_z[0]<<endl;

    /**======6、释放内存======*/
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_y));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}
