/**
https://blog.csdn.net/Bruce_0712/article/details/64928442

cudaDeviceSynchronize()：该方法将停止CPU端线程的执行，直到GPU端完成之前CUDA的任务，包括kernel函数、数据拷贝等。
cudaThreadSynchronize()：该方法的作用和cudaDeviceSynchronize()基本相同，但它不是一个被推荐的方法，也许在后期版本的CUDA中会被删除。
cudaStreamSynchronize()：这个方法接受一个stream ID，它将阻止CPU执行直到GPU端完成相应stream ID的所有CUDA任务，但其它stream中的CUDA任务可能执行完也可能没有执行完。

跨warp进行同步，您需要使用 __ syncthreads（）

(1)在同一个warp内的线程读写shared/global 不需同步,
  读写global和shared是立刻对本warp内的其他线程立刻可见的。

(2)在同一个block内的不同warp内线程读写shared/global 需同步,
    这种读写必须使用__syncthreads(), 或者__threadfence()来实现不同的读写可见效果。

(3)在同一个grid内的不同block内的线程读写shared/gloabl 需要同步
    这种读写必须使用__threadfence*()来实现一定的读写可见效果。


// 执行结果
[vec_add.cu:59] GPU 实现向量的加法
allocated 76.29 MB on GPU
GPU cost time=0.109505s	last_num=4e+07
CPU cost time=2.46585s	last_num=4e+07

*/
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <ctime>

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define use_warmup 1
using namespace std;
typedef float FLOAT;


// GPU预热(可以提升GPU计算速度)
void warmup();

// CPU 向量加法
// __host__ void vec_add_host(FLOAT *x,FLOAT* y,FLOAT *z,int N);
// or host函数__host__可以省略
 void vec_add_host(FLOAT *x,FLOAT *y,FLOAT *z,int N);


 // GPU 函数
 __global__ void vec_add(FLOAT *x,FLOAT *y,FLOAT *z,int N)
 {
     // 获取线程id（同时有很多个线程在执行这个函数，通过线程id区分）
     /**
     * <<<(256,256),256>>>  grid 2维 block 1维  tid=threadIdx.x+blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y
     * <<<256,256>>>  grid 1维 block 1维  tid=threadIdx.x+blockDim.x*blockIdx.x
     * <<<1,256>>>  grid 1维 block 1维  tid=threadIdx.x
     * <<<256,1>>>  grid 1维 block 1维  tid=blockDim.x*blockIdx.x
     */
     int tid=threadIdx.x+blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y;

     if (tid<N) z[tid]=x[tid]+y[tid]; // 开的线程数必须大于数据总数，保证每个数据都能参与计算
     // __syncthreads(); // 线程同步
 }


int main(int argc, char *argv[])
{
    mycout<<"GPU 实现向量的加法"<<endl;

    int N = 20000000;
    int nbytes = N * sizeof(FLOAT);

    /* 1D block */
    int bs = 256;

    /* 2D grid */
    // int s = ceil(sqrt((N + bs - 1.) / bs));
    int s = ceil(sqrt(1.0*N / bs));
    dim3 grid = dim3(s, s);

    // dx 表示gpu变量，hx表示cpu变量
    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL, *hy = NULL;
    FLOAT *dz = NULL, *hz = NULL;

    int itr = 30;
    int i;
    // double th, td;

    /**======1、CPU 创建变量赋值==========*/
    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    if (hx == NULL || hy == NULL || hz == NULL) {
        // printf("couldn't allocate CPU memory\n");
        mycout<<"couldn't allocate CPU memory"<<endl;
        return -2;
    }

    /*给CPU变量赋值*/
    // fill the arrays 'hx' and 'hy' on the CPU
    for (int i=0; i<N; i++) {
        hx[i] = i;
        hy[i] = i ;
    }

    /* warm up GPU */
    #if  use_warmup
    warmup();  // 预热
    #endif // use_warmup

    /**======2、GPU 分配内存======*/
    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    if (dx == NULL || dy == NULL || dz == NULL) {
        // printf("couldn't allocate GPU memory\n");
        mycout<<"couldn't allocate GPU memory"<<endl;
        return -1;
    }

    printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));

    /**======3、将CPU数据拷贝给GPU======*/
    /** copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    /* warm up GPU */
    #if  use_warmup
    warmup();  // 预热
    #endif // use_warmup

    /**======4、调用GPU计算======*/
    /* call GPU */
    // cudaThreadSynchronize(); // 线程同步，等到前面的GPU数据拷贝完成
    cudaDeviceSynchronize();  // cudaThreadSynchronize() 弃用了

    clock_t start = clock();
    for (i = 0; i < itr; i++) vec_add<<<grid, bs>>>(dx, dy, dz, N);
    // cudaThreadSynchronize(); // 线程同步 等待所有线程处理完成
    cudaDeviceSynchronize();  // cudaThreadSynchronize() 弃用了

    /**======5、GPU计算结果拷贝给CPU======*/
    cudaMemcpy(hz,dz, nbytes, cudaMemcpyDeviceToHost);

    cout<<"GPU cost time="<<(double)(clock()-start)/CLOCKS_PER_SEC <<"s\t"<<
    "last_num="<<hz[N-1]<<endl;

    // 计算CPU的时间
    start = clock();
    for (i = 0; i < itr; i++) vec_add_host(hx, hy, hz, N);
    cout<<"CPU cost time="<<(double)(clock()-start)/CLOCKS_PER_SEC <<"s\t"<<
    "last_num="<<hz[N-1]<<endl;

    // 释放内存
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);

    return 0;
}

/* warm up GPU */
__global__ void warmup_knl()
{
    int i, j;

    i = 1;
    j = 2;
    i = i + j;
}

void warmup()
{
    int i;

    for (i = 0; i < 8; i++) {
        warmup_knl<<<1, 256>>>();
    }
}

 void vec_add_host(FLOAT *x,FLOAT *y,FLOAT *z,int N)
 {
     for(int i=0;i<N;i++)
     {
         z[i]=x[i]+y[i];
     }
 }
