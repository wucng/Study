// nvcc main.cu -o main -I ../../common -std=c++11 -w

#include "common.h"

template <typename T>
__global__ void add(T *d_a,T *d_b,T *d_c,int n)
{
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if (idx>=n) return;
    d_c[idx]=d_a[idx]+d_b[idx];
}

template <typename T>
__global__
void global_matrix(T *dev_a,T *dev_b,T *dev_c,int height,int width)
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
void exec_add(T *d_a,T *d_b,T *d_c,int height,int width)
{
    dim3 block(nums_thread_pre_block,1,1);
    dim3 grid((height*width+nums_thread_pre_block-1)/nums_thread_pre_block,1,1);
    global_matrix<T><<<grid,block>>>(d_a,d_b,d_c,height,width);
    checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
}



template <typename T>
int getCostTime(Task<T> &task)
{
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    exec_add<T>(task.A,task.B,task.C,task.m,task.n);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

// 使用函数指针传递参数
template <typename T>
int getCostTime(void (*func)(T*,T*,T*,int, int) ,Task<T> &task)
{
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    func(task.A,task.B,task.C,task.m,task.n);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}


int main(int argc,char *argv[])
{   
    /**获取GPU的数量*/
    int deviceCount = 0;
    checkError(cudaGetDeviceCount(&deviceCount));
    mycout<<"GPU numbers:"<<deviceCount<<endl;
    if(deviceCount<1)
    {
        mycout<<"没有可用的GPU设备"<<endl;
        exit(-1);
    }

    if (argc>1)
    {
        int devID = findCudaDevice(argc, (const char **)argv);
        checkError(cudaSetDevice(devID));
        mycout<<"使用GPU："<<devID<<endl;
    }
    else
    {
        checkError(cudaSetDevice(0));//默认使用第一块GPU
        mycout<<"使用GPU："<<0<<endl;
    }

    srand((unsigned int)time(NULL)); //设置随机种子
    int height=0,width=0;
    height=max(5,rand()%10);
    width=max(5,rand()%10);
    Task<float> task;
    task.allocate2(height,width,1,0);

    getCostTime<float>(exec_add<float>,task);
    // getCostTime<float>(task);
    task.pprint();


    return 0;
}
