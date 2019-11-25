"""
使用pycuda加载权重文件，实现分类
"""
import numpy as np
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# 1、加载npz文件
parmas=np.load("models/tf_args.npz")
# print(list(parmas.keys()))

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def gpu_softmax(x,shape,dtype,threads_per_block=256):
    mod = SourceModule("""
          __global__ void gpu_exp(float *a_inOut,float *shape)
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
          
          __global__ void gpu_sum(float *a_in,float *a_out,float *shape)
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
          
          __global__ void gpu_div(float *a_inOut,float *a_sum,float *shape)
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
          
          """)

    # shape=x.shape
    # g_x = cuda.to_device(x)
    g_x = x
    g_x_sum=cuda.to_device(np.zeros([shape[0],],np.float32))

    block = (threads_per_block, 1, 1)
    grid = (shape[0]*shape[1] // threads_per_block + 1, 1, 1)

    func_gpu_exp=mod.get_function("gpu_exp")
    func_gpu_exp(g_x,cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    func_gpu_sum = mod.get_function("gpu_sum")
    func_gpu_sum(g_x,g_x_sum,cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    func_gpu_div = mod.get_function("gpu_div")
    func_gpu_div(g_x,g_x_sum, cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    # x = cuda.from_device(g_x, x.shape, x.dtype)

    return g_x,shape,dtype

def gpu_relu(x,shape,dtype,threads_per_block=256):
    mod = SourceModule("""
      __global__ void gpu_relu(float *a_in,float* shape)
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
      """)

    func = mod.get_function("gpu_relu")
    # g_x = cuda.to_device(x)
    g_x=x
    block=(threads_per_block,1,1)
    grid = (shape[0]*shape[1]//threads_per_block+1,1,1)
    func(g_x,cuda.to_device(np.asarray(shape,np.float32)), grid=grid, block=block)
    # x = cuda.from_device(g_x, x.shape, x.dtype)
    return g_x,shape,dtype

# 使用共享内存
def gpu_relu_SM(x,shape,dtype,threads_per_block=256):
    mod = SourceModule("""
      const int len_sm=256;
      __global__ void gpu_relu(float *a_in,float* shape)
      {
        //const int len_sm=blockDim.x; // 使用这个报错
        __shared__ float sdatas[len_sm]; // 使用共享内存
        const int N=(int)(shape[0]*shape[1]);
        int idx = threadIdx.x + blockIdx.x*blockDim.x; // 对应全局内存id
        int tid=threadIdx.x; // 对应共享内存id
    	if(idx>=N) return;
    	// 写入共享内存中
    	sdatas[tid]=a_in[idx]<0.0?0.0:a_in[idx];
    	__syncthreads();
    	// 共享内存写入全局内存
    	a_in[idx]=sdatas[tid];
      }
      """)

    func = mod.get_function("gpu_relu")
    # g_x = cuda.to_device(x)
    g_x=x
    block=(threads_per_block,1,1)
    grid = (shape[0]*shape[1]//threads_per_block+1,1,1)
    func(g_x,cuda.to_device(np.asarray(shape,np.float32)), grid=grid, block=block)
    # x = cuda.from_device(g_x, x.shape, x.dtype)
    return g_x,shape,dtype

def gpu_matrix_mul_add_bias(x,shape,wights,bias=None,threads_per_block=256):
    mod = SourceModule("""
          __global__ void gpu_matrix_mul(float *a_in,float *w,float *a_out,float *x_shape,float *w_shape)
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
          
          __global__ void gpu_matrix_add_vector(float *a_inOut,float *bias,float *shape)
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
          
          """)

    func_matrix_mul=mod.get_function("gpu_matrix_mul")
    # g_x=cuda.to_device(x)
    g_x=x
    g_w=cuda.to_device(wights)
    # x_shape=x.shape
    x_shape=shape
    w_shape=wights.shape
    tmp=np.zeros([x_shape[0],w_shape[1]],np.float32)
    g_tmp=cuda.to_device(tmp)
    grid=(x_shape[1]//threads_per_block+1,1,1)
    block=(threads_per_block,1,1)
    func_matrix_mul(g_x,g_w,g_tmp,cuda.to_device(np.asarray(x_shape,np.float32)),
                    cuda.to_device(np.asarray(w_shape,np.float32)),grid=grid, block=block)

    if bias is not None:
        g_b = cuda.to_device(bias)
        func_matrix_add_vector=mod.get_function("gpu_matrix_add_vector")
        grid = (tmp.size // threads_per_block + 1, 1, 1)
        # block = (threads_per_block, 1, 1)
        func_matrix_add_vector(g_tmp,g_b,cuda.to_device(np.asarray(tmp.shape,np.float32)),grid=grid, block=block)

    # tmp = cuda.from_device(g_tmp, tmp.shape, tmp.dtype)

    return g_tmp,tmp.shape,tmp.dtype


def model(parmas,img):
    """手动解析权重计算img的输出做分类"""
    x=np.reshape(img,[-1,28*28*1]).astype(np.float32)
    g_x = cuda.to_device(x)
    g_x,shape,dtype = gpu_matrix_mul_add_bias(g_x,x.shape, parmas['dense/kernel:0'], parmas['dense/bias:0'])
    g_x,shape,dtype =gpu_relu(g_x,shape,dtype)
    g_x,shape,dtype = gpu_matrix_mul_add_bias(g_x,shape,parmas['dense_1/kernel:0'],parmas['dense_1/bias:0'])

    # softmax
    g_x,shape,dtype = gpu_softmax(g_x,shape,dtype)
    # from GPU to cpu 最后一步才做从GPU 拷回CPU,不用每步都做
    x = cuda.from_device(g_x, shape, dtype)
    return x

start=time.time()
x_train, y_train, x_test, y_test=process_dataset()
print(np.argmax(model(parmas,x_test[:20]),1))
print(y_test[:20])
print(time.time()-start)
"""
[7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
0.3877
"""
