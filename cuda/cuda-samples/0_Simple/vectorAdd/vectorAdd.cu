/**
* Vector addition: C = A + B.
* nvcc vectorAdd.cu -I ../../common/ 
*/

#include "common.h"

template<typename T>
__device__ void add1(const T &a,const T &b,T &c)
{
    c=a+b;
}
// or
template<typename T>
__device__ T add2(const T &a,const T &b)
{
    T c=a+b;
    return c;
}

template<typename T>
__global__ void vectorAdd(const T *A,const T *B, T *C, const int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        // C[i] = A[i] + B[i];
        // add1<T>(A[i],B[i],C[i]);
        C[i]=add2<T>(A[i],B[i]);
    }
}

// 使用shared memory
template<typename T>
__global__ void shared_vectorAdd(const T *A,const T *B, T *C, const int numElements)
{
    extern __shared__ T sdatas[]; //共享内存声明
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    if(idx>=numElements) return;
    // global memory-->shared memory
    // sdatas[tid]=add2<T>(A[idx],B[idx]);
    sdatas[tid]=A[idx]+B[idx];
    __syncthreads(); // 必须线程同步，保证所有global内存变量都拷贝到共享内存

    // shared momery-->global momery
    C[idx]=sdatas[tid];
}

/**cpu数据传入，调用GPU运行*/
template<typename T>
int addInference(
    const T *h_a,
    const T *h_b,
    T *h_c,
    const int n // 数据总长度
    )
{
    size_t size=n*sizeof(T); // int size=n*sizeof(T);
    // 创建GPU变量并分配GPU内存
    T *d_a=NULL,*d_b=NULL,*d_c=NULL;
    checkError(cudaMalloc((void **)&d_a, size));
    checkError(cudaMalloc((void **)&d_b, size));
    checkError(cudaMalloc((void **)&d_c, size));

    // cpu-->GPU
    checkError(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));

    // 启动GPU计算
    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((n+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    cudaStream_t stream; // 创建流，如果不使用就是使用默认流
    checkError(cudaStreamCreate(&stream));

    // vectorAdd<<<grid,block,0,stream>>>(d_a,d_b,d_c,n); // global memory
    shared_vectorAdd<<<grid,block,nums_thread_pre_block*sizeof(T),stream>>>(d_a,d_b,d_c,n); // shared memory
    checkError(cudaGetLastError());// launch vectorAdd kernel

    // checkError(cudaDeviceSynchronize());//CPU等待GPU，使用多流时需使用

    // GPU-->CPU
    checkError(cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost));

    // free GPU
    checkError(cudaFree(d_a));
    checkError(cudaFree(d_b));
    checkError(cudaFree(d_c));

    // 销毁流
    checkError(cudaStreamDestroy(stream));

    return 0;
}

int main()
{
    myprint;
    const int N=50000;
    size_t size=N*sizeof(float);
    // CPU 
    float *h_a=NULL,*h_b=NULL,*h_c=NULL;
    // 分配内存 cudaMallocHost 比 malloc效率高
    checkError(cudaMallocHost((void**)&h_a,size));
    checkError(cudaMallocHost((void**)&h_b,size));
    checkError(cudaMallocHost((void**)&h_c,size));

    // 赋值
    // Initialize the host input vectors
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = rand()/(float)RAND_MAX;
        h_b[i] = rand()/(float)RAND_MAX;
    }

    // 调用函数执行
    addInference<float>(h_a,h_b,h_c,N);

    // 验证结果
    // Verify that the result vector is correct
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // free CPU
    // free(h_a); // 针对malloc分配空间时
    checkError(cudaFreeHost(h_a));
    checkError(cudaFreeHost(h_b));
    checkError(cudaFreeHost(h_c));

    return 0;
}
