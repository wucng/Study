#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>
#include "common/book.h"
#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)  // 2D grid,1D block

using namespace std;
typedef float FLOAT;

/*
void cpu_matrix_multiply_vector(FLOAT (*matrix_a)[width],
                                   FLOAT *vec_b,FLOAT *vec_c)
{
    FLOAT sum=0;
    for(int i=0;i<height;++i)
    {
        sum=0;
        for (int j=0;j<width;++j)
        {
            sum+=matrix_a[i][j]*vec_b[j];
        }
        vec_c[i]=sum;
    }
}
*/

__global__
void global_matrix(FLOAT *dev_a,FLOAT *dev_b,FLOAT *dev_c,const int height,const int width)
{
    int idx=get_tid();
    if(idx>=height*width) return;

    int rows=idx/width;
    int cols=idx%width;

    // 使用原子变量解决读写冲突
    // dev_c[rows]+=dev_a[idx]*dev_b[cols];
    atomicAdd(&dev_c[rows],dev_a[idx]*dev_b[cols]);
}

void global_matrix_multipy_vector(FLOAT *matrix_a,
                                  FLOAT *vec_b,FLOAT *vec_c,const int height,const int width)
{
    // GPU 变量
    FLOAT *dev_a=NULL,*dev_b=NULL,*dev_c=NULL;
    int nBytes=height*width*sizeof(FLOAT);
    int wBytes=width*sizeof(FLOAT);
    int hBytes=height*sizeof(FLOAT);

    HANDLE_ERROR(cudaMalloc((void **)&dev_a,nBytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b,wBytes));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c,hBytes));

    HANDLE_ERROR(cudaMemcpy(dev_a,matrix_a,nBytes,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b,vec_b,wBytes,cudaMemcpyHostToDevice));

    global_matrix<<<1,256>>>(dev_a,dev_b,dev_c,height,width);

    HANDLE_ERROR(cudaMemcpy(vec_c,dev_c,hBytes,cudaMemcpyDeviceToHost));

    // free
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
}


void pprint(FLOAT *vec_c,const int height,const int width)
{
    for(int i=0;i<height;++i)
    {
        for (int j=0;j<width;++j)
        {
            cout<< vec_c[j+i*width]<< " " ;
        }
        cout<<endl;
    }
}

int main()
{
    mycout<<"矩阵与向量相乘\n"<<
    "使用一维数组实现\n"<<
    "二维数组实践上是按一维数组方式存储"<<endl;

    const int height=3,width=5;
    FLOAT *matrix_a=NULL;
    int nBytes=height*width*sizeof(FLOAT);
    HANDLE_ERROR(cudaMallocHost((void **)&matrix_a,nBytes));

    // 赋值
    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            matrix_a[j+i*width]=1;
        }
    }

    int wBytes=width*sizeof(FLOAT);
    FLOAT *vec_b=NULL;
    HANDLE_ERROR(cudaMallocHost((void **)&vec_b,wBytes));
    for (int i=0;i<width;++i)
        vec_b[i]=1;

    int hBytes=height*sizeof(FLOAT);
    FLOAT *vec_c=NULL; // 存放结果
    HANDLE_ERROR(cudaMallocHost((void **)&vec_c,hBytes));

    {
        // 执行GPU计算
        global_matrix_multipy_vector(matrix_a,vec_b,vec_c,height,width);
    }

    // 打印结果
    pprint(vec_c,1,height);

    // free
    HANDLE_ERROR(cudaFreeHost(matrix_a));
    HANDLE_ERROR(cudaFreeHost(vec_b));
    HANDLE_ERROR(cudaFreeHost(vec_c));

    return 0;
}
