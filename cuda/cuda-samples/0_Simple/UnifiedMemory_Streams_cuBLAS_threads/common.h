#ifndef __VECTOR_ADD_H__
#define __VECTOR_ADD_H__

// #include "helper_cuda.h"
// #include "helper_functions.h"
#include <stdio.h>
#include <cuda.h>
#include <ctime>
// For the CUDA runtime routines (prefixed with "cuda_")
//#include <cuda_runtime.h>

// #include <iostream>
// using namespace std;
// #define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] " //<<endl
#define myprint printf("[%s:%d] ",__FILE__,__LINE__)
#define nums_thread_pre_block 256

static void HandleError(cudaError_t err,
                         const char *file,
                         int line )
{
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}

/**检测*/
#define checkError(err) (HandleError( err, __FILE__, __LINE__ ))


/**自定义数据*/
// 使用统一虚拟内存
template <typename T>
struct Task
{   
    unsigned int m=0,n=0,k=0,id=0; // A_mxn,B_nxk,C_mxk
    T *A=NULL,*B=NULL,*C=NULL;
    
    // 构造函数
    Task():m(0),n(0),k(0), id(0), A(NULL), B(NULL), C(NULL) {srand((unsigned int)time(NULL));};
    Task(unsigned int m,unsigned int n,unsigned int k) : m(m), n(n) ,k(k),id(0),  A(NULL), B(NULL), C(NULL)
    {
        // 使用统一虚拟内存(CPU与GPU都可以访问)
        // allocate unified memory -- the operation performed in this example will be a DGEMV
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n*k));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m*k));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    }

    ~Task()
    {
        // ensure all memory is deallocated
        checkError(cudaDeviceSynchronize());
        checkError(cudaFree(A));
        checkError(cudaFree(B));
        checkError(cudaFree(C));
    }

    void allocate2(const unsigned int m,const unsigned int n,const unsigned int k, const unsigned int unique_id)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        this->k = k;
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n*k));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m*k));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            A[i] = (T)(rand()%10);
        }

        for (int i=0; i<n*k; i++)
        {
            B[i] = (T)(rand()%10);
        }

        for (int i=0;i<m*k;++i)
        {
            C[i] = (T)0;
        }
    
    }

    void allocate(const unsigned int m,const unsigned int n, const unsigned int k,const unsigned int unique_id,
    cudaStream_t stream)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        this->k = k;
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n*k));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m*k));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            A[i] = 1;//drand48();
        }

        for (int i=0; i<n*k; i++)
        {
            B[i] = 1;//drand48();
        }

        for (int i=0;i<m*k;++i)
        {
            C[i] = 0;
        }
        // 异步拷贝到对应的流上处理
        // cudaMemAttachSingle单个流可以访问，cudaMemAttachHost CPU访问，cudaMemAttachGlobal 所有流都可以访问
        checkError(cudaStreamAttachMemAsync(stream, A, 0, cudaMemAttachSingle));
        checkError(cudaStreamAttachMemAsync(stream, B, 0, cudaMemAttachSingle));
        checkError(cudaStreamAttachMemAsync(stream, C, 0, cudaMemAttachSingle));
        checkError(cudaStreamSynchronize(stream));
        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    
    }

    void pprint()
    {
        myprint;printf("A:\n");
        for (int i=0; i<m; i++)
        {
            for(int j=0;j<n;++j)
            {
                printf("%.3f ",A[j+i*n]);
            }
            printf("\n");
        }
        printf("\n");

        myprint;printf("B:\n");
        for (int i=0; i<n; i++)
        {
            for(int j=0;j<k;++j)
            {
                printf("%.3f ",B[j+i*k]);
            }
            printf("\n");
        }
        printf("\n");

        myprint;printf("C:\n");
        for (int i=0; i<m; i++)
        {
            for(int j=0;j<k;++j)
            {
                printf("%.3f ",C[j+i*k]);
            }
            printf("\n");
        }
        printf("\n");
        
    }
};


#endif
