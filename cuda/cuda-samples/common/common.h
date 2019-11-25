#ifndef __VECTOR_ADD_H__
#define __VECTOR_ADD_H__

#include "helper_cuda.h"
#include "helper_functions.h"
#include <stdio.h>
#include <cuda.h>
#include <ctime>
// For the CUDA runtime routines (prefixed with "cuda_")
//#include <cuda_runtime.h>

#include <iostream>
using namespace std;
#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] " //<<endl
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
            A[i] = 1; //(T)(rand()%3);
        }

        for (int i=0; i<n*k; i++)
        {
            B[i] = 1; //(T)(rand()%3);
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

/************************************************************/
// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

template <typename T>
struct Data
{
    int flag=0; // 指定使用哪种CPU内存分配方式
    int nums=0;//数据长度
    T *h_a=NULL,*h_b=NULL,*h_c=NULL; // Pinned memory allocated on the CPU
    T *a_UA=NULL, *b_UA=NULL, *c_UA=NULL;  // Non-4K Aligned Pinned memory on the CPU
    T *d_a=NULL,*d_b=NULL,*d_c=NULL; // Device pointers for mapped memory
    Data():nums(0),flag(0),h_a(NULL),h_b(NULL),h_c(NULL),
                    d_a(NULL),d_b(NULL),d_c(NULL){srand((unsigned int)time(NULL));};

    /*
    Data(int nums,int flag):nums(nums),flag(flag),h_a(NULL),h_b(NULL),h_c(NULL),
                                d_a(NULL),d_b(NULL),d_c(NULL)
    {
        int bytes=nums*sizeof(T);
        // CPU空间分配
        #if(flag==0)
            // checkCudaErrors(cudaHostUnregister(h_a));
            checkCudaErrors(cudaHostRegister(h_a, bytes, cudaHostRegisterMapped));
            checkCudaErrors(cudaHostRegister(h_b, bytes, cudaHostRegisterMapped)); 
            checkCudaErrors(cudaHostRegister(h_c, bytes, cudaHostRegisterMapped)); 
        #elif(flag==1)
            // cudaFreeHost(h_a)
            checkCudaErrors(cudaHostAlloc((void **)&h_a, bytes, cudaHostAllocMapped));// or cudaMallocHost
            checkCudaErrors(cudaHostAlloc((void **)&h_b, bytes, cudaHostAllocMapped));
            checkCudaErrors(cudaHostAlloc((void **)&h_c, bytes, cudaHostAllocMapped));
        #elif(flag==2)
            // free cudaFreeHost(h_a)
            checkCudaErrors(cudaMallocHost((void **)&h_a, bytes)); 
            checkCudaErrors(cudaMallocHost((void **)&h_b, bytes));
            checkCudaErrors(cudaMallocHost((void **)&h_c, bytes));

        #else
            // free  free(h_a)
            h_a=(T *)malloc(bytes); 
            h_b=(T *)malloc(bytes);
            h_c=(T *)malloc(bytes);
        #endif
    }
    */
    ~Data()
    {
        if(flag==0)
        {
            checkCudaErrors(cudaHostUnregister(h_a));
            checkCudaErrors(cudaHostUnregister(h_b));
            checkCudaErrors(cudaHostUnregister(h_c));
            free(a_UA);
            free(b_UA);
            free(c_UA);
        }
 
        else if(flag==1)
        {
            checkCudaErrors(cudaFreeHost(h_a));
            checkCudaErrors(cudaFreeHost(h_b));
            checkCudaErrors(cudaFreeHost(h_c));
        }
            
        else if(flag==2)
        {
            checkCudaErrors(cudaFreeHost(h_a));
            checkCudaErrors(cudaFreeHost(h_b));
            checkCudaErrors(cudaFreeHost(h_c));
        }
            
        else
        {
            free(h_a);
            free(h_b);
            free(h_c);
        }

    }

    void allocate(int nums,int flag=0)
    {
        this->nums=nums;
        this->flag=flag;
        int bytes=nums*sizeof(T);
        // CPU空间分配
        if(flag==0)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaHostRegister"<<endl;
            a_UA = (T *) malloc(bytes + MEMORY_ALIGNMENT);
            b_UA = (T *) malloc(bytes + MEMORY_ALIGNMENT);
            c_UA = (T *) malloc(bytes + MEMORY_ALIGNMENT);

            // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
            h_a = (T *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
            h_b = (T *) ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
            h_c = (T *) ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

            // checkCudaErrors(cudaHostUnregister(h_a));
            checkCudaErrors(cudaHostRegister(h_a, bytes, cudaHostRegisterMapped));
            checkCudaErrors(cudaHostRegister(h_b, bytes, cudaHostRegisterMapped)); 
            checkCudaErrors(cudaHostRegister(h_c, bytes, cudaHostRegisterMapped));
        }
        
        else if(flag==1)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaHostAlloc"<<endl;
            // cudaFreeHost(h_a)
            checkCudaErrors(cudaHostAlloc((void **)&h_a, bytes, cudaHostAllocMapped));// or cudaMallocHost
            checkCudaErrors(cudaHostAlloc((void **)&h_b, bytes, cudaHostAllocMapped));
            checkCudaErrors(cudaHostAlloc((void **)&h_c, bytes, cudaHostAllocMapped));
        }
            

        else if(flag==2)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaMallocHost"<<endl;
            // free cudaFreeHost(h_a)
            checkCudaErrors(cudaMallocHost((void **)&h_a, bytes)); 
            checkCudaErrors(cudaMallocHost((void **)&h_b, bytes));
            checkCudaErrors(cudaMallocHost((void **)&h_c, bytes));
        }
            
        else
        {   
            // 这种方式分配无法使用cudaHostGetDevicePointer报错，必须使用（传统方式）
            mycout<<"flag = "<<flag<<" "<<"malloc"<<endl;
            // free  free(h_a)
            h_a=(T *)malloc(bytes); // C++方法
            h_b=(T *)malloc(bytes);
            h_c=(T *)malloc(bytes);
        }
            


        // CPU变量赋值
        for(int i=0;i<nums;++i)
        {
            h_a[i]=(T)(rand()%10);
            h_b[i]=(T)(rand()%10);
            h_c[i]=(T)0;
        }


        // cpu-->GPU(直接让GPU指针指向CPU指针，GPU与CPU都能访问，
        // 这样可以避免GPU内存分配以及CPU与GPU之间拷贝数据，比较类似于统一虚拟内存分配功能)
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)h_c, 0));

        /** 
        // 与上面等价 （传统方式）
        // 分配GPU内存
        checkCudaErrors(cudMalloc((void **)&d_a,bytes));
        checkCudaErrors(cudMalloc((void **)&d_b,bytes));
        checkCudaErrors(cudMalloc((void **)&d_c,bytes));
        // cpu-->GPU
        checkCudaErrors(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_c,h_c,bytes,cudaMemcpyHostToDevice));
        */
    }

    void pprint()
    {
        mycout<<"---------h_a------------"<<endl;
        for(int i=0;i<nums;++i)
            cout<<h_a[i]<<" ";
        cout<<endl;

        mycout<<"---------h_b------------"<<endl;
        for(int i=0;i<nums;++i)
            cout<<h_b[i]<<" ";
        cout<<endl;

        mycout<<"---------h_c------------"<<endl;
        for(int i=0;i<nums;++i)
            cout<<h_c[i]<<" ";
        cout<<endl;
    }
};

template <typename T>
struct Data2d
{
    int flag=0; // 指定使用哪种CPU内存分配方式
    int m=0,n=0,k=0;//数据长度
    T *h_a=NULL,*h_b=NULL,*h_c=NULL; // Pinned memory allocated on the CPU
    T *a_UA=NULL, *b_UA=NULL, *c_UA=NULL;  // Non-4K Aligned Pinned memory on the CPU
    T *d_a=NULL,*d_b=NULL,*d_c=NULL; // Device pointers for mapped memory
    Data2d():m(0),n(0),k(0),flag(0),h_a(NULL),h_b(NULL),h_c(NULL),
                    d_a(NULL),d_b(NULL),d_c(NULL){srand((unsigned int)time(NULL));};


    ~Data2d()
    {
        if(flag==0)
        {
            checkCudaErrors(cudaHostUnregister(h_a));
            checkCudaErrors(cudaHostUnregister(h_b));
            checkCudaErrors(cudaHostUnregister(h_c));
            free(a_UA);
            free(b_UA);
            free(c_UA);
        }
 
        else if(flag==1)
        {
            checkCudaErrors(cudaFreeHost(h_a));
            checkCudaErrors(cudaFreeHost(h_b));
            checkCudaErrors(cudaFreeHost(h_c));
        }
            
        else if(flag==2)
        {
            checkCudaErrors(cudaFreeHost(h_a));
            checkCudaErrors(cudaFreeHost(h_b));
            checkCudaErrors(cudaFreeHost(h_c));
        }
            
        else
        {
            free(h_a);
            free(h_b);
            free(h_c);
        }

    }

    void allocate(int m,int n,int k,int flag=0)
    {
        this->m=m;
        this->n=n;
        this->k=k;
        this->flag=flag;
        int a_bytes=m*n*sizeof(T);
        int b_bytes=n*k*sizeof(T);
        int c_bytes=m*k*sizeof(T);

        // CPU空间分配
        if(flag==0)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaHostRegister"<<endl;
            a_UA = (T *) malloc(a_bytes + MEMORY_ALIGNMENT);
            b_UA = (T *) malloc(b_bytes + MEMORY_ALIGNMENT);
            c_UA = (T *) malloc(c_bytes + MEMORY_ALIGNMENT);

            // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
            h_a = (T *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
            h_b = (T *) ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
            h_c = (T *) ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

            // checkCudaErrors(cudaHostUnregister(h_a));
            checkCudaErrors(cudaHostRegister(h_a, a_bytes, cudaHostRegisterMapped));
            checkCudaErrors(cudaHostRegister(h_b, b_bytes, cudaHostRegisterMapped)); 
            checkCudaErrors(cudaHostRegister(h_c, c_bytes, cudaHostRegisterMapped));
        }
        
        else if(flag==1)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaHostAlloc"<<endl;
            // cudaFreeHost(h_a)
            checkCudaErrors(cudaHostAlloc((void **)&h_a, a_bytes, cudaHostAllocMapped));// or cudaMallocHost
            checkCudaErrors(cudaHostAlloc((void **)&h_b, b_bytes, cudaHostAllocMapped));
            checkCudaErrors(cudaHostAlloc((void **)&h_c, c_bytes, cudaHostAllocMapped));
        }
            

        else if(flag==2)
        {
            mycout<<"flag = "<<flag<<" "<<"cudaMallocHost"<<endl;
            // free cudaFreeHost(h_a)
            checkCudaErrors(cudaMallocHost((void **)&h_a, a_bytes)); 
            checkCudaErrors(cudaMallocHost((void **)&h_b, b_bytes));
            checkCudaErrors(cudaMallocHost((void **)&h_c, c_bytes));
        }
            
        else
        {   
            // 这种方式分配无法使用cudaHostGetDevicePointer报错，必须使用（传统方式）
            mycout<<"flag = "<<flag<<" "<<"malloc"<<endl;
            // free  free(h_a)
            h_a=(T *)malloc(a_bytes); // C++方法
            h_b=(T *)malloc(b_bytes);
            h_c=(T *)malloc(c_bytes);
        }
            


        // CPU变量赋值
        for (int i=0;i<m*n;++i)
            h_a[i]=1;//(T)(rand()%10);
        
        for (int i=0;i<n*k;++i)
            h_b[i]=1;//(T)(rand()%10);
        
        for (int i=0;i<m*k;++i)
            h_c[i]=0;//(T)(rand()%10);

        // cpu-->GPU(直接让GPU指针指向CPU指针，GPU与CPU都能访问，
        // 这样可以避免GPU内存分配以及CPU与GPU之间拷贝数据，比较类似于统一虚拟内存分配功能)
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)h_c, 0));

        /** 
        // 与上面等价 （传统方式）
        // 分配GPU内存
        checkCudaErrors(cudMalloc((void **)&d_a,bytes));
        checkCudaErrors(cudMalloc((void **)&d_b,bytes));
        checkCudaErrors(cudMalloc((void **)&d_c,bytes));
        // cpu-->GPU
        checkCudaErrors(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_c,h_c,bytes,cudaMemcpyHostToDevice));
        */
    }

    void pprint()
    {
        mycout<<"---------h_a------------"<<endl;
        for(int i=0;i<m;++i)
        {
            for (int j=0;j<n;++j)
            {
                cout<<h_a[i*n+j]<<" ";
            }
            cout<<endl;
        } 
        cout<<endl;

        mycout<<"---------h_b------------"<<endl;
        for(int i=0;i<n;++i)
        {
            for (int j=0;j<k;++j)
            {
                cout<<h_b[i*k+j]<<" ";
            }
            cout<<endl;
        } 
        cout<<endl;

        mycout<<"---------h_c------------"<<endl;
        for(int i=0;i<m;++i)
        {
            for (int j=0;j<k;++j)
            {
                cout<<h_c[i*k+j]<<" ";
            }
            cout<<endl;
        } 
        cout<<endl;
    }
};

#endif
