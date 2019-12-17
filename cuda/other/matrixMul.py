"""
# 分块处理 每次处理 32x32 的块
matrixMul(A,B)  # 14.27755355834961 (ms)

# 分段处理 每次处理 32 的段（将A的一行与B的一列按32为一段切分处理）
matrixMul3(A,B)  # 23.067617416381836 (ms)

# 使用共享内存，A的一列(B的一行)为共享内存大小
matrixMul2(A,B,True) # 249.05755519866943 (ms)

# 使用全局内存，使用原子加法
matrixMul2(A,B,False) # 124.63619709014893 (ms)

numpy 矩阵相乘  # 1.3933420181274414 (ms)

# 共享内存设置太大反而会影响速度，如果太长可以分成多段处理（多块处理）
# 当数据量增加 matrixMul(A,B)效率会接近 numpy 矩阵相乘，甚至超过
"""


import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
import time

# 设置使用哪块GPU
cuda.Device(0)

def matrixMul(A:np.array,B:np.array)->np.array:
    """
    C=A*B
    A:float32
    B:float32
    """
    mod=SourceModule(
        """ 
        __global__ void matrix_mul(float *A,float *B,float *C,int *shape)
        {   
            const int BLOCK_SIZE=32; // 块大小 32x32
            
            // Block index
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // Thread index
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // 每个block的所有线程id
            int tid = tx+ty*blockDim.x;
            
            // int bid = bx+by*gridDim.x;
            // 全局所有线程id
            // int idx = bid*(blockDim.x*blockDim.y) + tid;
            
            float Csub = 0.0f; // 每个线程的局部变量
            int idxa=0,idxb=0,idxc=0;
            
            int A_cols = shape[1]/BLOCK_SIZE;
            
            for(int c=0;c<A_cols;++c)
            {
                // 每次读取一块做矩阵乘法
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; // 共享变量声明并定义
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
                
                idxa = tid + (c+by*gridDim.x)*(blockDim.x*blockDim.y);
        
                idxb = tid + (bx+c*gridDim.x)*(blockDim.x*blockDim.y);
                
                // 全局内存写入共享内存
                As[ty][tx] = A[idxa];
                Bs[ty][tx] = B[idxb];
                
                // Synchronize to make sure the matrices are loaded
                __syncthreads();
                
                // 每个block内的线程依次做，不会有线程冲突
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    Csub += As[ty][j] * Bs[j][tx]; // 并不会冲突
                }
                __syncthreads();                
            }
            // each thread writes one element
            idxc = tid + (bx+by*gridDim.x)*(blockDim.x*blockDim.y);
            C[idxc] = Csub;
        }
        """
    )

    A_shape = A.shape # h,w 格式
    B_shape = B.shape
    assert A_shape[1] == B_shape[0],print("failure")

    # cpu-->GPU
    g_A = cuda.to_device(A)
    g_B = cuda.to_device(B)
    C = np.zeros([A_shape[0],B_shape[1]],A.dtype) # 用于保存结果
    g_C = cuda.to_device(C)

    g_shape = cuda.to_device(np.asarray(A_shape,np.int32))

    func = mod.get_function("matrix_mul")

    block_size =32
    # 让block 对应共享内存的大小
    block = (block_size,block_size,1) # x,y,z格式 x对应w，y对应h
    # grid = ((A_shape[0]+block[0]-1)//block[0],(B_shape[1]+block[0]-1)//block[1],1)
    grid = ((B_shape[1]+block[0]-1)//block[0],(A_shape[0]+block[1]-1)//block[1],1)

    # 执行核函数
    func(g_A,g_B,g_C,g_shape,grid=grid,block=block)

    # GPU -->CPU
    C = cuda.from_device(g_C,C.shape,C.dtype)

    return C


def matrixMul2(A: np.array, B: np.array,isSharedMemory=False) -> np.array:
    """
    C=A*B
    A:float32
    B:float32
    """
    mod = SourceModule(
        """ 
        __global__ void matrix_mul(float *A,float *B,float *C)
        {   
            int tx = threadIdx.x;
            
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // int A_h = gridDim.y;
            int A_w = blockDim.x;
            // int B_h = blockDim.x;
            int B_w = gridDim.x;
            
            // C[bx+by*B_w] += A[tx+by*A_w]*B[bx+tx*B_w];
            // atomicAdd(&C[bx+by*B_w] ,A[tx+by*A_w]*B[bx+tx*B_w]); // 这个原子同步会占用大部分时间
            atomicAdd_system(&C[bx+by*B_w] ,A[tx+by*A_w]*B[bx+tx*B_w]); 
        }
        
        // 使用共享内存，减少原子同步操作
        __global__ void matrix_mul_shared(float *A,float *B,float *C)
        {   
            // 每次处理A一行与B一列
            const int BLOCK_SIZE = 32 * 16;
            __shared__ float As[BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE];
            
            int tx = threadIdx.x;
            
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // int A_h = gridDim.y;
            int A_w = blockDim.x;
            // int B_h = blockDim.x;
            int B_w = gridDim.x;
            
            // 全局内存 到共享内存
            As[tx] = A[tx+by*A_w];
            Bs[tx] = B[bx+tx*B_w];
            float Csum=0.0f;
            for(int i=0;i<A_w;++i)
            {
                Csum+=As[i]*Bs[i];
            }
            
            C[bx+by*B_w] = Csum;           
        }
        
        """
    )

    A_shape = A.shape  # h,w 格式
    B_shape = B.shape
    assert A_shape[1] == B_shape[0], print("failure")

    # cpu-->GPU
    g_A = cuda.to_device(A)
    g_B = cuda.to_device(B)
    C = np.zeros([A_shape[0], B_shape[1]], A.dtype)  # 用于保存结果
    g_C = cuda.to_device(C)

    # g_shape = cuda.to_device(np.asarray(A_shape, np.int32))

    if isSharedMemory:
        func = mod.get_function("matrix_mul_shared")
    else:
        func = mod.get_function("matrix_mul")

    block = (A_shape[1], 1, 1)  # x,y,z格式 x对应w，y对应h
    grid = (B_shape[1],A_shape[0],1)

    # 执行核函数
    func(g_A, g_B, g_C, grid=grid, block=block)

    # GPU -->CPU
    C = cuda.from_device(g_C, C.shape, C.dtype)

    return C


def matrixMul3(A: np.array, B: np.array) -> np.array:
    """
    C=A*B
    A:float32
    B:float32
    """
    mod = SourceModule(
        """ 
        // 使用共享内存，减少原子同步操作
        __global__ void matrix_mul_shared(float *A,float *B,float *C,int *shape)
        {   
            // 每次处理A一行与B一列 (分段处理)
            const int BLOCK_SIZE = 32;

            int tx = threadIdx.x;

            int bx = blockIdx.x;
            int by = blockIdx.y;

            // int A_h = gridDim.y;
            int A_w = shape[1];
            // int B_h = blockDim.x;
            int B_w = gridDim.x;

            float Csum=0.0f;
            for(int c=0;c<A_w/BLOCK_SIZE;++c)
            {
                __shared__ float As[BLOCK_SIZE];
                __shared__ float Bs[BLOCK_SIZE];

                // 全局内存 到共享内存
                As[tx] = A[tx+by*A_w];
                Bs[tx] = B[bx+tx*B_w];
                
                for(int i=0;i<BLOCK_SIZE;++i)
                {
                    Csum+=As[i]*Bs[i];
                }
            }
            
            C[bx+by*B_w] = Csum;           
        }
        """
    )

    A_shape = A.shape  # h,w 格式
    B_shape = B.shape
    assert A_shape[1] == B_shape[0], print("failure")

    # cpu-->GPU
    g_A = cuda.to_device(A)
    g_B = cuda.to_device(B)
    C = np.zeros([A_shape[0], B_shape[1]], A.dtype)  # 用于保存结果
    g_C = cuda.to_device(C)

    g_shape = cuda.to_device(np.asarray(A_shape, np.int32))

    func = mod.get_function("matrix_mul_shared")
    block_size = 32
    # block = (A_shape[1], 1, 1)  # x,y,z格式 x对应w，y对应h
    block = (block_size, 1, 1)  # x,y,z格式 x对应w，y对应h
    grid = (B_shape[1], A_shape[0], 1)

    # 执行核函数
    func(g_A, g_B, g_C,g_shape, grid=grid, block=block)

    # GPU -->CPU
    C = cuda.from_device(g_C, C.shape, C.dtype)

    return C


def main():
    A = np.ones([32*16, 32 * 16]).astype(np.float32) # 必须是float32
    B = np.ones([32 * 16, 32 * 16]).astype(np.float32) # 必须是float32

    times=10
    start = time.time()
    for i in range(times):
        C=matrixMul(A,B)
        # C=matrixMul2(A,B,True)
        # C=matrixMul3(A,B)
    print((time.time()-start)/times*1000,"(ms)")

    # 使用numpy
    start = time.time()
    for i in range(times):
        # C = matrixMul(A, B)
        C2 = np.matmul(A, B)
    print((time.time() - start) / times*1000,"(ms)")

    # 计算error
    # err=np.max(np.fabs(C-C2))
    err=np.max((C-C2)**2)

    print(err)


def main2():
    A = np.ones([32 * 16, 32 * 16]).astype(np.float32)  # 必须是float32
    B = np.ones([32 * 16, 32 * 16]).astype(np.float32)  # 必须是float32
    C = matrixMul2(A, B)
    C2 = np.matmul(A, B)

    err = np.max((C - C2) ** 2)
    print(err)

if __name__ == '__main__':
    main()
