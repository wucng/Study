/**
* 使用cuda自定义，调用出错？参考：matrix_mul.cu（已经实现）
* 1.使用统一虚拟内存(CPU与GPU都可以访问，这样就不需要CPU与GPU之间拷贝) cudaMallocManaged
* 2.使用cudaStream_t （多流）
* 3.使用cuBLAS 成功
// * 4.使用多线程 pthread，推荐使用thread简单点 未实现 参考：matrix_mul.cu（已经实现）
* 5.实现矩阵乘以向量(C_m=A_mxn * B_n)
* 6.编译：nvcc simple.cu -o simple -I ./ -lcublas -lpthread -std=c++11 -w
* 7.执行：./simple
*/

#include "common.h"
// #include <cstdio>
// #include <ctime>
// #include <vector>
// #include <algorithm>
// #include <pthread.h> // 只能Linux使用
// #include <thread> // Linux与windows都可以使用

// #include <stdlib.h>

// cuBLAS
#include <cublas_v2.h> // 这个会和 using namespace std;冲突
// #include "cuda_runtime.h"


// #define FLOAT double
typedef double FLOAT;

// void srand48(long seed)
// {
//     srand((unsigned int)seed);
// }
// FLOAT drand48()
// {
//     // return (FLOAT)rand()/RAND_MAX;
//     return (FLOAT)(rand()%3);
// }


template <typename T>
__global__
void global_matrix(T *dev_a,T *dev_b,T *dev_c, int height, int width)
{
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx>=height*width) return;

    int rows=idx/width;
    int cols=idx%width;

    // 使用原子变量解决读写冲突
    // dev_c[rows]+=dev_a[idx]*dev_b[cols];
    atomicAdd(&dev_c[rows],dev_a[idx]*dev_b[cols]);
}


template <typename T>
void gpu_gemv(T *matrix_a,
              T *vec_b,T *vec_c, int height, int width)
{    
    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    global_matrix<T><<<grid,block>>>(matrix_a,vec_b,vec_c,height,width);
    checkError(cudaGetLastError());// launch vectorAdd kernel
}


template <typename T>
struct Method
{
    // CPU执行
    // simple host dgemv: assume data is in row-major format and square
    void gemv(const int &m, const int &n, const T &alpha, const T *A, const T *x, const T &beta, T *result)
    {
        // A_mxn;x_n ;result_m
        // result=alpha*A*x+beta
        // rows
        for (int i=0; i<m; i++)
        {   
            // result[i] *= beta;
            result[i]=(T)0;

            for (int j=0; j<n; j++)
            {
                result[i] += A[i*n+j]*x[j];
            }
            result[i]*=alpha;
            result[i]+=beta;
        }
    }

    // lcublas
    void cublas_gemv(cublasHandle_t handle,int &m, int &n,
    T &alpha, const T *A,const T *x, T &beta, T *result) 
    {
        int k=1;
        cudaDeviceSynchronize();
        cublasDgemv(handle, CUBLAS_OP_T, //CUBLAS_OP_N,//CUBLAS_OP_C,
        n,m, &alpha, A, n , x, 1, &beta, result, 1);
        // cublasDgemm(handle, //cublasSgemm
        // CUBLAS_OP_N,
        // CUBLAS_OP_N,
        // k,m,n,&alpha,
        // x,k,A,n,&beta, result,k
        // );
        cudaDeviceSynchronize();
    }

    
    void gpu_gemv(T *A,
              T *x,T *result,const int height,const int width)
    {    
        dim3 block(nums_thread_pre_block,1,1);
        dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
        global_matrix<T><<<grid,block>>>(A,x,result,height,width);
        checkError(cudaGetLastError());// launch vectorAdd kernel
        cudaDeviceSynchronize(); //等待GPU执行完成，(使用统一虚拟内存时必须加上）
    }
};


int main()
{
    
    Task<FLOAT> *t=new Task<FLOAT>();
    Method<FLOAT> *modus=new Method<FLOAT>();

    int m=5,n=2,id=0;
    FLOAT alpha=(FLOAT)1.0;
    FLOAT beta=(FLOAT)0.0;

    t->allocate2(m,n,1,id);
    // modus->gemv(m,n,alpha,t->A,t->B,beta,t->C);
    // modus->gpu_gemv(t->A,t->B,t->C,m,n);

    cublasHandle_t handles;
    cublasCreate(&handles);
    modus->cublas_gemv(handles,m,n,alpha,t->A,t->B,beta,t->C);

    t->pprint();

    // free
    cublasDestroy(handles);
    delete t;
    delete modus;

    return 0;
}