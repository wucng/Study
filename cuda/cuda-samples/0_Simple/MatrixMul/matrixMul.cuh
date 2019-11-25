#include "common.h"
typedef float FLOAT;

/*****************************无法实用****************************************************/
/**数据放到1个block里运行,且满足m=n=k=BLOCK_SIZE<=32 ,>32共享内存出问题？ 无法实用*/
template <int BLOCK_SIZE> 
__global__ 
void MatrixMulCUDA0(FLOAT *A,FLOAT *B,FLOAT *C,unsigned int m,unsigned int n,unsigned int k)
{
    /**
    * A mxn
    * B nxk
    * C nxk
    */
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    // FLOAT Csub = 0; // 每个线程的局部变量

    // 全局变量都写入到共享内存中(共享内存 对应每个block内的线程id)
    __shared__ FLOAT As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ FLOAT Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ FLOAT Cs[BLOCK_SIZE][BLOCK_SIZE];

    if (idx>=BLOCK_SIZE*BLOCK_SIZE) return;

    As[ty][tx] = A[idx];
    Bs[ty][tx] = B[idx];
    Cs[ty][tx] = 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // 每个block内做矩阵乘法,每个block没32个线程组成1个warp，warp内的线程会自动同步，因此BLOCK_SIZE必须小于等于32
    // 如果超过了 则必须使用原子加操作
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        // Csub += As[ty][j] * Bs[j][tx];
        // atomicAdd(&Csub,As[ty][j] * Bs[j][tx]);
        atomicAdd(&Cs[ty][tx],As[ty][j] * Bs[j][tx]);
    }
    __syncthreads();

    // each thread writes one element
    // C[idx] = Csub;
    C[idx] = Cs[ty][tx];

}


/**数据放到1个block里运行,且满足m=n=k=BLOCK_SIZE<=32  无法实用*/
template <int BLOCK_SIZE> 
__global__ 
void MatrixMulCUDA1(FLOAT *A,FLOAT *B,FLOAT *C,unsigned int m,unsigned int n,unsigned int k)
{
    /**
    * A mxn
    * B nxk
    * C nxk
    */
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    FLOAT Csub = 0; // 每个线程的局部变量

    // 全局变量都写入到共享内存中(共享内存 对应每个block内的线程id)
    __shared__ FLOAT As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ FLOAT Bs[BLOCK_SIZE][BLOCK_SIZE];

    if (idx>=BLOCK_SIZE*BLOCK_SIZE) return;

    As[ty][tx] = A[idx];
    Bs[ty][tx] = B[idx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // 每个block内做矩阵乘法,每个block没32个线程组成1个warp，warp内的线程会自动同步，因此BLOCK_SIZE必须小于等于32
    // 如果超过了 则必须使用原子加操作
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        Csub += As[ty][j] * Bs[j][tx];
        // atomicAdd(&Csub,As[ty][j] * Bs[j][tx]);
    }
    __syncthreads();

    // each thread writes one element
    C[idx] = Csub;

}
/*****************************************矩阵乘法***************************************/

/**跨block运行,且满足WA%BLOCK_SIZE=0  WB%BLOCK_SIZE=0  
BLOCK_SIZE<=32 将数据按 BLOCK_SIZE x BLOCK_SIZE 切成多个块循环执行
时间复杂度 O(m/32*k/32*m/32)
每次只执行1个block
*/
template <int BLOCK_SIZE> 
__global__ 
void MatrixMulSelf(FLOAT *A,FLOAT *B,FLOAT *C,unsigned int m,unsigned int n,unsigned int k)
{
    /**
    * A mxn
    * B nxk
    * C nxk
    */
    // Block index
    // int bx = blockIdx.x;
    // int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    // int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    int idxa=0,idxb=0,idxc=0;
    
    for(int a=0;a<m/BLOCK_SIZE;++a)
    {      
        for (int b=0;b<k/BLOCK_SIZE;++b)
        {
            FLOAT Csub = 0; // 每个线程的局部变量

            for(int c=0;c<n/BLOCK_SIZE;++c)
            {
                
                // 全局变量都写入到共享内存中(共享内存 对应每个block内的线程id)
                __shared__ FLOAT As[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ FLOAT Bs[BLOCK_SIZE][BLOCK_SIZE];

                idxa = tid + (c+a*gridDim.x)*(blockDim.x*blockDim.y);
                
                idxb = tid + (b+c*gridDim.x)*(blockDim.x*blockDim.y);

                As[ty][tx] = A[idxa];
                Bs[ty][tx] = B[idxb];

                // Synchronize to make sure the matrices are loaded
                __syncthreads();

                // 每个block内做矩阵乘法,每个block没32个线程组成1个warp，warp内的线程会自动同步，因此BLOCK_SIZE必须小于等于32
                // 如果超过了 则必须使用原子加操作
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    Csub += As[ty][j] * Bs[j][tx]; // 并不会冲突
                }
                __syncthreads();                
            }
            // each thread writes one element
            idxc = tid + (b+a*gridDim.x)*(blockDim.x*blockDim.y);
            C[idxc] = Csub;
        }      
    }
}


// ------------------------推荐使用-------------------------------------------
/**跨block运行,且满足WA%BLOCK_SIZE=0  WB%BLOCK_SIZE=0  
BLOCK_SIZE<=32 将数据按 BLOCK_SIZE x BLOCK_SIZE 切成多个块循环执行
时间复杂度 O(k/32)
每次执行所有block
*/
template <int BLOCK_SIZE> 
__global__ 
void MatrixMulSelfV2(FLOAT *A,FLOAT *B,FLOAT *C,unsigned int m,unsigned int n,unsigned int k)
{
    /**
    * A mxn
    * B nxk
    * C nxk
    */
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    // int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    int idxa=0,idxb=0,idxc=0;
    
    FLOAT Csub = 0; // 每个线程的局部变量

    for(int c=0;c<n/BLOCK_SIZE;++c)
    {
        
        // 全局变量都写入到共享内存中(共享内存 对应每个block内的线程id)
        __shared__ FLOAT As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ FLOAT Bs[BLOCK_SIZE][BLOCK_SIZE];

        idxa = tid + (c+by*gridDim.x)*(blockDim.x*blockDim.y);
        
        idxb = tid + (bx+c*gridDim.x)*(blockDim.x*blockDim.y);

        As[ty][tx] = A[idxa];
        Bs[ty][tx] = B[idxb];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // 每个block内做矩阵乘法,每个block没32个线程组成1个warp，warp内的线程会自动同步，因此BLOCK_SIZE必须小于等于32
        // 如果超过了 则必须使用原子加操作
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            Csub += As[ty][j] * Bs[j][tx]; // 并不会冲突
        }
        __syncthreads();                
    }
    // each thread writes one element
    idxc = tid + (bx+by*gridDim.x)*(blockDim.x*blockDim.y);
    C[idxc] = Csub;

}



/**跨block运行,且满足WA%BLOCK_SIZE=0  WB%BLOCK_SIZE=0  
BLOCK_SIZE<=32 将数据按 BLOCK_SIZE x BLOCK_SIZE 切成多个块循环执行
时间复杂度 O(m/32*k/32)
*/
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *A,
                                                        float *B, 
                                                        float *C,
                                                        int wA, // A的行
                                                        int wB) // B的列
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


/**矩阵乘法 时间复杂度 O(k)*/
template <typename T>
__global__ void gpu_matrix_mul(T *dev_a,T *dev_b,T *dev_c,unsigned int m,unsigned int n,unsigned int k)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // int i =(int)(idx/col); // 行索引
    // int j =(int)(idx%col); // 列索引

    if (idx>=m*n) return;

    int rows=idx/n;
    int cols=idx%n;

    for(int i=0;i<k;i+=1) // 循环dev_b的列  每次按dev_a x dev_b的一列（矩阵与向量相乘）
    {
        // 使用原子变量解决读写冲突
        // dev_c[rows]+=dev_a[idx]*dev_b[cols];
        atomicAdd(&dev_c[rows*k+i],dev_a[idx]*dev_b[cols*k+i]);
        // atomicAdd(&dev_c[rows*k+i+1],dev_a[idx]*dev_b[cols*k+i+1]);
        // atomicAdd(&dev_c[rows*k+i+2],dev_a[idx]*dev_b[cols*k+i+2]);
        // atomicAdd(&dev_c[rows*k+i+3],dev_a[idx]*dev_b[cols*k+i+3]);
    }
    
}


/**矩阵乘法 时间复杂度 O(m*k)*/
template <typename T>
__global__ void gpu_matrix_mul1(T *a_in,T *w,T *a_out,unsigned int m,unsigned int n,unsigned int k)
{
    // const int N_x = (int)(x_shape[0]*x_shape[1]);
    // const int N_w = (int)(w_shape[0]*w_shape[1]);
    const int x_row = m;
    const int x_col = n;

    const int w_row = n;
    const int w_col = k;

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


/*********************************矩阵乘以向量*************************************************************/
/* 矩阵乘以向量 */
/***使用全局内存，所有线程作为计算单元***/
template <typename T>
__global__ 
void matrix_mul_vector(T *dev_a,T *dev_b,T *dev_c,const int height,const int width)
{   
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    // int idx=get_tid();
    if(idx>=height*width) return;

    int rows=idx/width;
    int cols=idx%width;

    // 使用原子变量解决读写冲突
    // dev_c[rows]+=dev_a[idx]*dev_b[cols];
    atomicAdd(&dev_c[rows],dev_a[idx]*dev_b[cols]);
    // or
    //atomicAdd_system(&dev_c[rows],dev_a[idx]*dev_b[cols]);
}

// nvcc test.cu -I ../../common/  -std=c++11 -w  // -arch compute_75 -code sm_75
/**使用shared memory每个block作为计算单元*/
template <int SIZE>
__global__
void matrix_mul_vector_sm(FLOAT *dev_a,FLOAT *dev_b,FLOAT *dev_c,const int height,const int width)
{   
    __shared__ FLOAT As[SIZE]; // 读取矩阵dev_a的每一行
    __shared__ FLOAT Bs[SIZE]; // 读取向量dev_b的所有列
    // extern __shared__ T Cs[]; // 保存向量结果
    __shared__ FLOAT Cs[1]; // 保存每个block计算结果

    // block id
    int bid = blockIdx.x;
    // 每个block内thread id
    int tid = threadIdx.x;
    // 全局线程id
    int idx = tid + bid * blockDim.x;

    if(idx>=height*width) return;

    // FLOAT Csum = 0;
    // 全局内存-->共享内存
    As[tid] = dev_a[idx];
    Bs[tid] = dev_b[tid];
    Cs[0] = 0;
    __syncthreads();

    // 以每个block为计算单元(每个block最后结果加和只有一个)
    // atomicAdd_block(&Cs[0],As[tid]*Bs[tid]); // atomicAdd_block不会跨block
    atomicAdd(&Cs[0],As[tid]*Bs[tid]); // 由于读取的数据都是来在共享内存，共享内存是不能跨block，因此能保证是同一个block内部计算
                                      // 如果读取的数据都是来在全局内存，全局内存是能跨block，因此是所有block内的线程操作
    __syncthreads();

    //写入全局内存
    if(tid==0) // 因为只要写一次，因此使用一个线程来写
        dev_c[bid]=Cs[tid];
}


//参考矩阵乘法 分块计算 矩阵乘向量
template <int SIZE>
__global__
void matrix_mul_vector_block(FLOAT *dev_a,FLOAT *dev_b,FLOAT *dev_c,const int height,const int width)
{
    // __shared__ FLOAT As[SIZE][SIZE]; // 读取矩阵dev_a的一块
    // __shared__ FLOAT Bs[SIZE]; // 读取向量dev_b的一块
    // extern __shared__ T Cs[]; // 保存向量结果
    // __shared__ FLOAT Cs[1]; // 保存每个block计算结果

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个block的所有线程id
    int tid = tx+ty*blockDim.x;

    // 全局所有线程id
    int idx = (bx+by*gridDim.x)*(blockDim.x*blockDim.y) + tid;

    if(idx>=height*width) return;

    int idxa=0,idxb=0,idxc=0;
    FLOAT Csub = 0;// 每个线程的局部变量
    for(int c=0;c<width/SIZE;++c)
    {
        __shared__ FLOAT As[SIZE][SIZE]; // 读取矩阵dev_a的一块
        __shared__ FLOAT Bs[SIZE]; // 读取向量dev_b的一块

        idxa = tid + (c+by*gridDim.x)*(blockDim.x*blockDim.y);
        // idxb = tid + (bx+c*gridDim.x)*(blockDim.x*blockDim.y);
        idxb = ty + c*blockDim.y;

        // 全局内存-->共享内存
        As[ty][tx] = dev_a[idxa];
        Bs[ty] = dev_b[idxb];
        __syncthreads();

        // 每个block内做矩阵乘法,每个block没32个线程组成1个warp，warp内的线程会自动同步，因此BLOCK_SIZE必须小于等于32
        // 如果超过了 则必须使用原子加操作
        for (int j = 0; j < SIZE; ++j) {
            Csub += As[ty][j] * Bs[j]; // 并不会冲突
        }
        __syncthreads();                

    }
    // each thread writes one element
    idxc = ty + by*blockDim.y;
    dev_c[idxc] = Csub;
}