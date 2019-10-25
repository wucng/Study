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

__global__ void global_pooling(FLOAT *dev_x,FLOAT *dev_z,int height,int width)
{
    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    // int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    // int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    if(idx>=height*width ) return; // 超出边界
    // if(idx>=(height-1)*(width-1) ) return; // 超出边界

    int row=idx/width;
    int col=idx%width;

    if(row%2!=0 || col %2!=0) return;

    FLOAT img_value=0;
    int cur_row=0;
    int cur_col=0;

    FLOAT tmp_value=-1.0f;

    for(int i=0;i<2;++i)
    {
        for(int j=0;j<2;++j)
        {
            cur_row=row+i;
            cur_col=col+j;

            if(cur_row<0 || cur_col<0 || cur_row>=height || cur_col>=width)
            {
                img_value=0;
            }
            else
            {
                img_value=dev_x[cur_row*width+cur_col];
            }

            // dev_z[cur_row/2*width/2+cur_col/2]+=img_value*1/4; // 平均池化

            // 最大池化
            tmp_value=tmp_value<img_value?img_value:tmp_value;
        }
    }

    // only for 最大池化
    dev_z[cur_row/2*width/2+cur_col/2]=tmp_value;
}

__global__ void shared_pooling(FLOAT *dev_x,FLOAT *dev_z,int height,int width)
{
    extern __shared__ FLOAT sdatas[];// 声明共享内存

    int idx = get_tid(); // 全局索引，针对全局内存访问 （以每个线程为计算单元）
    int tid = get_tid_per_block();  //每个block中threads的Idx，针对每个block的共享内存访问 （以每个block为计算单元（以每个block内所有线程组为计算单元））
    // int bid = get_bid(); // block的索引（存储每个block中间结果的访问索引）（每个block的索引）

    if(idx>=height*width) return; // 超出边界

    // 每个block将对应的全局变量复制到各自的共享内存(每个block都有一个独立的共享内存)
    sdatas[tid]=idx<height*width?dev_x[idx] : 0;
	__syncthreads();

    int row=idx/width;
    int col=idx%width;

    // dev_z[col*height+row]=dev_x[idx];
    dev_z[col*height+row]=sdatas[tid];

}

int main()
{
    mycout<<"2维池化操作- 2x2 池化"<<endl;
    const int height=4; // image 高
    const int width= 4; // image 宽

    // 把二维展开成一维 （数组内存都是按一维排列）
    int nbytes=height*width*sizeof(FLOAT);
    int nhalfbytes=height/2*width/2*sizeof(FLOAT);

    FLOAT *host_x=NULL,*host_z=NULL;
    FLOAT *dev_x=NULL,*dev_z=NULL;

    /**1D block*/
    int bs=1024;

    /**1D grid*/
    int grid=ceil(1.0*height*width/bs);

    // 分配内存
    HANDLE_ERROR(cudaMallocHost((void **)&host_x, nbytes));
    HANDLE_ERROR(cudaMallocHost((void **)&host_z, nhalfbytes));

    HANDLE_ERROR(cudaMalloc((void **)&dev_x, nbytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_z, nhalfbytes));

    // 设置随机种子
    srand((unsigned int) time(NULL) );

    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            host_x[j+i*width]= rand()%256; // [0,255) 随机数
            // host_x[j+i*width]= 1;
        }
    }

    // CPU-->GPU
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, nbytes, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaDeviceSynchronize()); // CPU 等待GPU操作完成

    // CPU 启动 GPU kernel计算
    {
        // 使用全局变量
        global_pooling<<<grid,bs>>>(dev_x,dev_z,height,width);

        // 使用共享内存变量 (共享内存变量只能block内部通信，跨block没法访问)
        // shared_pooling<<<grid,bs,bs*sizeof(FLOAT)>>>(dev_x,dev_z,height,width);
    }

    // GPU ---> CPU
    HANDLE_ERROR(cudaMemcpy(host_z, dev_z, nhalfbytes, cudaMemcpyDeviceToHost));

    // 打印部分结果
    mycout<<"池化前"<<endl;

    for(int i=0;i<height*width;++i)
    {
        cout<<host_x[i]<<" ";
    }
    cout<<endl;

    mycout<<"池化后"<<endl;
     for(int i=0;i<height/2*width/2;++i)
    {
        cout<<host_z[i]<<" ";
    }
    cout<<endl;

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_z));

    // cudaMallocHost 释放方式
    HANDLE_ERROR(cudaFreeHost(host_x));
    HANDLE_ERROR(cudaFreeHost(host_z));

    return 0;
}
