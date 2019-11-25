// nvcc matrix_mul.cu -o simple -I ../../common -std=c++11 -w
/**
* 0.使用普通的内存分配 cudaMallocHost，cudaMalloc
* 1.使用统一虚拟内存(CPU与GPU都可以访问，这样就不需要CPU与GPU之间拷贝) cudaMallocManaged
* 2.使用cudaStream_t （多流）
* 4.使用多线程 pthread，推荐使用thread简单点
* 5.实现矩阵乘以向量(C_m=A_mxn * B_n)
* 6.编译：nvcc matrix_mul.cu -o simple -I ../../common -lpthread -std=c++11 -w  //-lcublas
* 7.执行：./simple
*/

#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>
#include <thread> // 使用多线程
#include <cmath>
// #include <cublas_v2.h>
// #include "common/book.h"
#include "common.h"
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

void srand48(long seed)
{
    srand((unsigned int)seed);
}
double drand48()
{
    // return (double)rand()/RAND_MAX;
    return (double)(rand()%3);
}

template <typename T>
__global__
void global_matrix(T *dev_a,T *dev_b,T *dev_c,const int height,const int width)
{
    int idx=get_tid();
    if(idx>=height*width) return;

    int rows=idx/width;
    int cols=idx%width;

    // 使用原子变量解决读写冲突
    // dev_c[rows]+=dev_a[idx]*dev_b[cols];
    atomicAdd(&dev_c[rows],dev_a[idx]*dev_b[cols]);
}


template <typename T>
void gpu_gemv(T *matrix_a,
              T *vec_b,T *vec_c,const int height,const int width)
{    
    cout<<"thread id:"<<this_thread::get_id()<<endl;

    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    global_matrix<T><<<grid,block>>>(matrix_a,vec_b,vec_c,height,width);
    checkError(cudaGetLastError());// launch vectorAdd kernel
    cudaDeviceSynchronize(); //等待GPU执行完成，(使用统一虚拟内存时必须加上）

    return;
}

template <typename T>
void gpu_gemv_thread(T *matrix_a,
              T *vec_b,T *vec_c,const int height,const int width,cudaStream_t stream)
{    
    cout<<"thread id:"<<this_thread::get_id()<<endl;

    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    global_matrix<T><<<grid,block,0,stream>>>(matrix_a,vec_b,vec_c,height,width);
    checkError(cudaGetLastError());// launch vectorAdd kernel
    cudaDeviceSynchronize(); //等待GPU执行完成，(使用统一虚拟内存时必须加上）

    return;
}


template <typename T>
void global_matrix_multipy_vector(T *matrix_a,
                                  T *vec_b,T *vec_c,const int height,const int width)
{
    // GPU 变量
    T *dev_a=NULL,*dev_b=NULL,*dev_c=NULL;
    int nBytes=height*width*sizeof(T);
    int wBytes=width*sizeof(T);
    int hBytes=height*sizeof(T);

    checkError(cudaMalloc((void **)&dev_a,nBytes));
    checkError(cudaMalloc((void **)&dev_b,wBytes));
    checkError(cudaMalloc((void **)&dev_c,hBytes));

    checkError(cudaMemcpy(dev_a,matrix_a,nBytes,cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(dev_b,vec_b,wBytes,cudaMemcpyHostToDevice));

    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    global_matrix<T><<<grid,block>>>(dev_a,dev_b,dev_c,height,width);
    // gpu_gemv<T>(dev_a,dev_b,dev_c,height,width);

    checkError(cudaMemcpy(vec_c,dev_c,hBytes,cudaMemcpyDeviceToHost));

    // free
    checkError(cudaFree(dev_a));
    checkError(cudaFree(dev_b));
    checkError(cudaFree(dev_c));
}


// 使用统一虚拟内存
template <typename T>
struct Task
{   
    unsigned int m=0,n=0,id=0;
    T *A=NULL,*B=NULL,*C=NULL;
    
    // 构造函数
    Task():m(0),n(0), id(0), A(NULL), B(NULL), C(NULL) {srand48(time(NULL));};
    Task(unsigned int m,unsigned int n) : m(m), n(n),id(0),  A(NULL), B(NULL), C(NULL)
    {
        // 使用统一虚拟内存(CPU与GPU都可以访问)
        // allocate unified memory -- the operation performed in this example will be a DGEMV
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m));
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

    void allocate2(const unsigned int m,const unsigned int n, const unsigned int unique_id)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            A[i] = 1;//drand48();
        }

        for (int i=0; i<n; i++)
        {
            B[i] = 1;//drand48();
        }

        for (int i=0;i<m;++i)
        {
            C[i] = (FLOAT)0;
        }
    
    }

    void allocate(const unsigned int m,const unsigned int n, const unsigned int unique_id,
    cudaStream_t stream)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        checkError(cudaMallocManaged((void **)&A, sizeof(T)*m*n));
        checkError(cudaMallocManaged((void **)&B, sizeof(T)*n));
        checkError(cudaMallocManaged((void **)&C, sizeof(T)*m));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            A[i] = 1;//drand48();
        }

        for (int i=0; i<n; i++)
        {
            B[i] = 1;//drand48();
        }

        for (int i=0;i<m;++i)
        {
            C[i] = (FLOAT)0;
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
            printf("%.3f ",B[i]);
        }
        printf("\n");

        myprint;printf("C:\n");
        for (int i=0;i<m;++i)
            printf("%.3f ",C[i]);
        
        printf("\n");
        
    }
};


// 不使用统一虚拟内存
template <typename T>
struct TaskSecond
{   
    unsigned int m=0,n=0,id=0;
    T *h_A=NULL,*h_B=NULL,*h_C=NULL;
    T *d_A=NULL,*d_B=NULL,*d_C=NULL;
    
    // 构造函数
    TaskSecond():m(0),n(0), id(0), h_A(NULL), h_B(NULL), h_C(NULL), d_A(NULL), d_B(NULL), d_C(NULL){srand48(time(NULL));};
    TaskSecond(unsigned int m,unsigned int n) : m(m), n(n),id(0),h_A(NULL), h_B(NULL), 
                                                h_C(NULL), d_A(NULL), d_B(NULL), d_C(NULL)
    {
        // 分配CPU内存
        checkError(cudaMallocHost((void **)&h_A, sizeof(T)*m*n));
        checkError(cudaMallocHost((void **)&h_B, sizeof(T)*n));
        checkError(cudaMallocHost((void **)&h_C, sizeof(T)*m));

        // 分配GPU内存
        checkError(cudaMalloc((void **)&d_A, sizeof(T)*m*n));
        checkError(cudaMalloc((void **)&d_B, sizeof(T)*n));
        checkError(cudaMalloc((void **)&d_C, sizeof(T)*m));

        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    }

    ~TaskSecond()
    {
        // ensure all memory is deallocated
        // checkError(cudaDeviceSynchronize());
        // free CPU
        checkError(cudaFreeHost(h_A));
        checkError(cudaFreeHost(h_B));
        checkError(cudaFreeHost(h_C));
        // free GPU
        checkError(cudaFree(d_A));
        checkError(cudaFree(d_B));
        checkError(cudaFree(d_C));
    }

    // 使用默认流
    void allocate2(const unsigned int m,const unsigned int n, const unsigned int unique_id)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        // 分配CPU内存
        checkError(cudaMallocHost((void **)&h_A, sizeof(T)*m*n));
        checkError(cudaMallocHost((void **)&h_B, sizeof(T)*n));
        checkError(cudaMallocHost((void **)&h_C, sizeof(T)*m));

        // 分配GPU内存
        checkError(cudaMalloc((void **)&d_A, sizeof(T)*m*n));
        checkError(cudaMalloc((void **)&d_B, sizeof(T)*n));
        checkError(cudaMalloc((void **)&d_C, sizeof(T)*m));

        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            h_A[i] = 1;//drand48();
        }

        for (int i=0; i<n; i++)
        {
            h_B[i] = 1;//drand48();
        }

        for (int i=0;i<m;++i)
        {
            h_C[i] = (FLOAT)0;
        }

        checkError(cudaMemcpy(d_A, h_A, sizeof(T)*m*n, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(d_B, h_B, sizeof(T)*n, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(d_C, h_C, sizeof(T)*m, cudaMemcpyHostToDevice));
        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    }

    // 指定流
    void allocate(const unsigned int m,const unsigned int n, const unsigned int unique_id,
    cudaStream_t stream)
    {
        // allocate unified memory outside of constructor
        this->id = unique_id;
        this->m = m;
        this->n = n;
        // 分配CPU内存
        checkError(cudaMallocHost((void **)&h_A, sizeof(T)*m*n));
        checkError(cudaMallocHost((void **)&h_B, sizeof(T)*n));
        checkError(cudaMallocHost((void **)&h_C, sizeof(T)*m));

        // 分配GPU内存
        checkError(cudaMalloc((void **)&d_A, sizeof(T)*m*n));
        checkError(cudaMalloc((void **)&d_B, sizeof(T)*n));
        checkError(cudaMalloc((void **)&d_C, sizeof(T)*m));

        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        // srand48(time(NULL));
        // srand((unsigned int)time(NULL));

        // populate data with random elements
        for (int i=0; i<m*n; i++)
        {
            h_A[i] = 1;//drand48();
        }

        for (int i=0; i<n; i++)
        {
            h_B[i] = 1;//drand48();
        }

        for (int i=0;i<m;++i)
        {
            h_C[i] = (FLOAT)0;
        }
        // 异步拷贝到对应的流上处理
        checkError(cudaMemcpyAsync(d_A, h_A, sizeof(T)*m*n, cudaMemcpyHostToDevice,stream));
        checkError(cudaMemcpyAsync(d_B, h_B, sizeof(T)*n, cudaMemcpyHostToDevice,stream));
        checkError(cudaMemcpyAsync(d_C, h_C, sizeof(T)*m, cudaMemcpyHostToDevice,stream));
        // checkError(cudaStreamSynchronize(stream));
        checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成    
    }


    // 将GPU的结果拷贝回CPU
    void copyG2C(cudaStream_t stream)
    {
        checkError(cudaMemcpyAsync(h_C,d_C, sizeof(T)*m, cudaMemcpyDeviceToHost,stream));
    }

    void copyG2C2()
    {
        checkError(cudaMemcpy(h_C,d_C,sizeof(T)*m, cudaMemcpyDeviceToHost));
    }


    void pprint()
    {
        myprint;printf("A:\n");
        for (int i=0; i<m; i++)
        {
            for(int j=0;j<n;++j)
            {
                printf("%.3f ",h_A[j+i*n]);
            }
            printf("\n");
        }
        printf("\n");

        myprint;printf("B:\n");
        for (int i=0; i<n; i++)
        {
            printf("%.3f ",h_B[i]);
        }
        printf("\n");

        myprint;printf("C:\n");
        for (int i=0;i<m;++i)
            printf("%.3f ",h_C[i]);
        
        printf("\n");
        
    }
};


template <typename T>
void pprint2(T *vec_c,const int height,const int width)
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

template <typename T>
void pprint(T *A,T *B,T *C,int m,int n)
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
        printf("%.3f ",B[i]);
    }
    printf("\n");

    myprint;printf("C:\n");
    for (int i=0;i<m;++i)
        printf("%.3f ",C[i]);
    
    printf("\n");
    
}

template <typename T>
void exec_task2(TaskSecond<T> &task)
{
    gpu_gemv<T>(task.d_A,task.d_B,task.d_C,task.m,task.n);
    task.copyG2C2();
}

template <typename T>
void exec_task(TaskSecond<T> &task,cudaStream_t stream)
{
    gpu_gemv_thread<T>(task.d_A,task.d_B,task.d_C,task.m,task.n,stream);
    task.copyG2C(stream);
}

template <typename T>
void exec_tast_thred(T *d_A,T *d_B,T *d_C,T *h_C,int height,int width,cudaStream_t stream)
{
    gpu_gemv_thread<T>(d_A,d_B,d_C,height,width,stream);
    checkError(cudaMemcpyAsync(h_C,d_C, sizeof(T)*height, cudaMemcpyDeviceToHost,stream));
}

template <typename T>
void exec_tast_thred2(T *d_A,T *d_B,T *d_C,T *h_C,int height,int width)
{
    gpu_gemv<T>(d_A,d_B,d_C,height,width);
    checkError(cudaMemcpy(h_C,d_C, sizeof(T)*height, cudaMemcpyDeviceToHost));
}



int main()
{
    /**获取GPU的数量*/
    int deviceCount = 0;
    checkError(cudaGetDeviceCount(&deviceCount));
    mycout<<"GPU numbers:"<<deviceCount<<endl;
    if(deviceCount<1) exit(-1);
    mycout<<"使用哪块GPU：0"<<endl;
    checkError(cudaSetDevice(0));//设置使用哪块GPU

    const int NUMS_THREAD=2;
    thread t[NUMS_THREAD]; // 创建2个线程数组   
    TaskSecond<FLOAT>* tasks=new TaskSecond<FLOAT>[NUMS_THREAD]();
    cudaStream_t *streams = new cudaStream_t[NUMS_THREAD]();
    int height=0,width=0;
    for(int i=0;i<NUMS_THREAD;++i)
    {
        checkError(cudaStreamCreate(&streams[i]));
        height=max(5,rand()%10);
        width=max(5,rand()%10);

        {
            // 没有使用线程
            // tasks[i].allocate2(height,width,i);
            // exec_task2<FLOAT>(tasks[i]);
            // or
            // tasks[i].allocate(height,width,i,streams[i]);
            // exec_task<FLOAT>(tasks[i],streams[i]);
        }
        
        {
            // 使用线程
            // tasks[i].allocate2(height,width,i);
            // t[i]=thread(exec_task2<FLOAT>,tasks[i]); // 报错
            // t[i]=thread(exec_tast_thred2<FLOAT>,tasks[i].d_A,tasks[i].d_B,tasks[i].d_C,tasks[i].h_C,height,width);

            tasks[i].allocate(height,width,i,streams[i]);
            // t[i]=thread(exec_task<FLOAT>,tasks[i],streams[i]); // 报错
            t[i]=thread(exec_tast_thred<FLOAT>,tasks[i].d_A,tasks[i].d_B,tasks[i].d_C,tasks[i].h_C,height,width,streams[i]);

            if(t[i].joinable()) t[i].join();
        }

        
    }

    // print
    for(int i=0;i<NUMS_THREAD;++i)
    {
        tasks[i].pprint();
    }

    // free
    delete[] tasks;

    for(int i=0;i<NUMS_THREAD;++i)
        cudaStreamDestroy(streams[i]);

    delete[] streams;

    return 0;
}


int main00()
{
    const int NUMS_THREAD=2;
    thread t[NUMS_THREAD]; // 创建2个线程数组   
    Task<FLOAT>* tasks=new Task<FLOAT>[NUMS_THREAD]();
    cudaStream_t *streams = new cudaStream_t[NUMS_THREAD]();
    int height=0,width=0;
    for(int i=0;i<NUMS_THREAD;++i)
    {
        checkError(cudaStreamCreate(&streams[i]));
        height=max(5,rand()%10);
        width=max(5,rand()%10);
        
        // tasks[i].allocate2(height,width,i);
        // t[i]=thread(gpu_gemv<FLOAT>,tasks[i].A,tasks[i].B,tasks[i].C,height,width);
        // gpu_gemv<FLOAT>(tasks[i].A,tasks[i].B,tasks[i].C,height,width);

        tasks[i].allocate(height,width,i,streams[i]);
        t[i]=thread(gpu_gemv_thread<FLOAT>,tasks[i].A,tasks[i].B,tasks[i].C,height,width,streams[i]);
        if(t[i].joinable()) t[i].join();

        // thread tp(gpu_gemv_thread<FLOAT>,tasks[i].A,tasks[i].B,tasks[i].C,height,width,streams[i]);
        // if(tp.joinable()) tp.join();
    }

    // print
    for(int i=0;i<NUMS_THREAD;++i)
    {
        tasks[i].pprint();
    }

    // free
    delete[] tasks;

    for(int i=0;i<NUMS_THREAD;++i)
        cudaStreamDestroy(streams[i]);

    delete[] streams;

    return 0;
}


int main01()
{
    unsigned int m=5,n=3,id=0;
    Task<FLOAT> t;
    t.allocate2(m,n,id);
    // cudaStream_t *streams = new cudaStream_t[1];
    // cublasHandle_t *handles = new cublasHandle_t[1];

    // attach managed memory to my stream
    // cublasSetStream(handles[0], streams[0]);
    // attach managed memory to a (dummy) stream to allow host access while the device is running
    // checkError(cudaStreamAttachMemAsync(streams[0], t.A, 0, cudaMemAttachSingle));
    // checkError(cudaStreamAttachMemAsync(streams[0], t.B, 0, cudaMemAttachSingle));
    // checkError(cudaStreamAttachMemAsync(streams[0], t.C, 0, cudaMemAttachSingle));
    // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
    // checkError(cudaStreamSynchronize(streams[0]));

    gpu_gemv<FLOAT>(t.A,t.B,t.C,m,n);

    t.pprint();

    // cudaStreamDestroy(streams[0]);
    // cublasDestroy(handles[0]);

    return 0;

}



int main02()
{
    mycout<<"矩阵与向量相乘\n"<<
    "使用一维数组实现\n"<<
    "二维数组实践上是按一维数组方式存储"<<endl;

    const int height=3,width=5;
    FLOAT *matrix_a=NULL;
    int nBytes=height*width*sizeof(FLOAT);
    checkError(cudaMallocHost((void **)&matrix_a,nBytes));

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
    checkError(cudaMallocHost((void **)&vec_b,wBytes));
    for (int i=0;i<width;++i)
        vec_b[i]=1;

    int hBytes=height*sizeof(FLOAT);
    FLOAT *vec_c=NULL; // 存放结果
    checkError(cudaMallocHost((void **)&vec_c,hBytes));

    {
        // 执行GPU计算
        global_matrix_multipy_vector<FLOAT>(matrix_a,vec_b,vec_c,height,width);
    }

    // 打印结果
    pprint<FLOAT>(matrix_a,vec_b,vec_c,height,width);

    // free
    checkError(cudaFreeHost(matrix_a));
    checkError(cudaFreeHost(vec_b));
    checkError(cudaFreeHost(vec_c));

    return 0;
}

int main03()
{
    mycout<<"矩阵与向量相乘\n"<<
    "使用一维数组实现\n"<<
    "二维数组实践上是按一维数组方式存储"<<endl;

    const int height=3,width=5;
    FLOAT *matrix_a=NULL;
    int nBytes=height*width*sizeof(FLOAT);
    checkError(cudaMallocManaged((void **)&matrix_a,nBytes));

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
    checkError(cudaMallocManaged((void **)&vec_b,wBytes));
    for (int i=0;i<width;++i)
        vec_b[i]=1;

    int hBytes=height*sizeof(FLOAT);
    FLOAT *vec_c=NULL; // 存放结果
    checkError(cudaMallocManaged((void **)&vec_c,hBytes));

    {
        // 执行GPU计算
        gpu_gemv<FLOAT>(matrix_a,vec_b,vec_c,height,width);
    }

    // 打印结果
    pprint<FLOAT>(matrix_a,vec_b,vec_c,height,width);

    // free
    checkError(cudaFree(matrix_a));
    checkError(cudaFree(vec_b));
    checkError(cudaFree(vec_c));

    return 0;
}
