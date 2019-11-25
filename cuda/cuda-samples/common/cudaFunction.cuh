/**定义cuda 核函数*/
#ifndef __CUDA_FUNCTION_CUH__
#define __CUDA_FUNCTION_CUH__


/* 向量相加 */
template <typename T>
__global__ void vectorAddGPU(T *a, T *b, T *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // if (idx>=N) return;

    // if (idx < N)
    // {
    //     c[idx] = a[idx] + b[idx];
    // }
    while(idx<N)
    {
        c[idx] = a[idx] + b[idx];
        idx+=blockDim.x*gridDim.x;
    }
}


/* 矩阵相加 */
template <typename T>
__global__ void matrixAddGPU(T *a, T *b, T *c, int height,int width)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    while(idx<height*width)
    {
        c[idx] = a[idx] + b[idx];
        idx+=blockDim.x*gridDim.x;
    }
}


/* 矩阵乘以向量 */
template <typename T>
__global__
void matrix_mul_vector(T *dev_a,T *dev_b,T *dev_c,const int height,const int width)
{
    int idx=get_tid();
    if(idx>=height*width) return;

    int rows=idx/width;
    int cols=idx%width;

    // 使用原子变量解决读写冲突
    // dev_c[rows]+=dev_a[idx]*dev_b[cols];
    atomicAdd(&dev_c[rows],dev_a[idx]*dev_b[cols]);
    // or
    //atomicAdd_system(&dev_c[rows],dev_a[idx]*dev_b[cols]);
}



/***-----------softmax-----------------**********/
/**/
template <typename T>
__global__ void gpu_exp(T *a_inOut,T *shape)
{
    const int N = (int)(shape[0]*shape[1]);
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // if(idx>=N) return;
    while(idx<N)
    {
        a_inOut[idx]=expf(a_inOut[idx]);
        idx+=blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void gpu_sum(T *a_in,T *a_out,T *shape)
{
    // 每行求加和
    const int N = (int)(shape[0]*shape[1]);
    // const int row=(int)shape[0];
    const int col=(int)shape[1];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    
    // 使用原子算法求加和
    while(idx<N)
    {
        atomicAdd(&a_out[(int)(idx/col)],a_in[idx]);
        idx+=blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void gpu_div(T *a_inOut,T *a_sum,T *shape)
{
    const int N = (int)(shape[0]*shape[1]);
    // const int row=(int)shape[0];
    const int col=(int)shape[1];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    while(idx<N)
    {
        a_inOut[idx]/=a_sum[(int)(idx/col)];
        idx+=blockDim.x * gridDim.x;
    }
}
/**--------------------------------------------------------*/
/**------relu---------*/
template <typename T>
__global__ void gpu_relu(T *a_in,T* shape)
{
    const int N=(int)(shape[0]*shape[1]);
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // if(idx>=N) return;
    while(idx<N)
    {
        a_in[idx]=a_in[idx]<0.0?0.0:a_in[idx];
        idx+=blockDim.x * gridDim.x;
    }
}

/**矩阵乘法*/
template <typename T>
__global__ void gpu_matrix_mul(T *a_in,T *w,T *a_out,T *x_shape,T *w_shape)
{
    // const int N_x = (int)(x_shape[0]*x_shape[1]);
    // const int N_w = (int)(w_shape[0]*w_shape[1]);
    const int x_row=(int)x_shape[0];
    const int x_col=(int)x_shape[1];

    const int w_row=(int)w_shape[0];
    const int w_col=(int)w_shape[1];

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // int i =(int)(idx/col); // 行索引
    // int j =(int)(idx%col); // 列索引

    if (idx>=x_col || x_col!=w_row) return;

    /*
    // 每次处理一行与一列
    for(int i=0;i<x_row;++i)
    {
        for(int j=0;j<w_col;++j)
        {
        atomicAdd(&a_out[j+i*w_col],a_in[idx+i*x_col]*w[j+idx*w_col]);
        }
    }
    */


    // 每次处理多行与多列(按块处理) // 每次处理两行两列
    for(int i=0;i<x_row;i+=2)
    {
        for(int j=0;j<w_col;j+=2)
        {                
        atomicAdd(&a_out[j+i*w_col],a_in[idx+i*x_col]*w[j+idx*w_col]);// i,j
        atomicAdd(&a_out[j+(i+1)*w_col],a_in[idx+(i+1)*x_col]*w[j+idx*w_col]);// i+1,j
        atomicAdd(&a_out[j+1+i*w_col],a_in[idx+i*x_col]*w[j+1+idx*w_col]);// i,j+1
        atomicAdd(&a_out[j+1+(i+1)*w_col],a_in[idx+(i+1)*x_col]*w[j+1+idx*w_col]);// i+1,j+1
        }
    }

}

/**矩阵与向量相加*/
template <typename T>
__global__ void gpu_matrix_add_vector(T *a_inOut,T *bias,T *shape)
{
    const int N = (int)(shape[0]*shape[1]);
    // const int row=(int)shape[0];
    const int col=(int)shape[1];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // if(idx>=N) return;
    while(idx<N)
    {
        a_inOut[idx]+=bias[idx%col];
        idx+=blockDim.x * gridDim.x;
    }
}


#endif