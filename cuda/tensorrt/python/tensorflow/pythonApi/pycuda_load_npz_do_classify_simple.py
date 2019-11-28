"""
使用pycuda加载权重文件，实现分类
"""
import numpy as np
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
from pycuda.driver import Stream
import time
from threading import Thread

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

def gpu_softmax(x, threads_per_block=256):
    mod = SourceModule("""
          __global__ void gpu_exp(float *a_inOut,int *shape)
          {
            const int N = shape[0]*shape[1];
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
        	// if(idx>=N) return;
        	while(idx<N)
            {
              a_inOut[idx]=expf(a_inOut[idx]);
              idx+=blockDim.x * gridDim.x;
            }
          }

          __global__ void gpu_sum(float *a_in,float *a_out,int *shape)
          {
             // 每行求加和
             const int N = shape[0]*shape[1];
             // const int row=shape[0];
             const int col=shape[1];
             int idx = threadIdx.x + blockIdx.x*blockDim.x;

             // 使用原子算法求加和
             while(idx<N)
            {
              atomicAdd(&a_out[idx/col],a_in[idx]);
              idx+=blockDim.x * gridDim.x;
            }
          }

          __global__ void gpu_div(float *a_inOut,float *a_sum,int *shape)
          {
             const int N = shape[0]*shape[1];
             // const int row=shape[0];
             const int col=shape[1];
             int idx = threadIdx.x + blockIdx.x*blockDim.x;
             while(idx<N)
            {
              a_inOut[idx]/=a_sum[idx/col];
              idx+=blockDim.x * gridDim.x;
            }
          }

          """)

    shape = x.shape
    g_x = cuda.to_device(x)
    g_x_sum = cuda.to_device(np.zeros([x.shape[0], ], np.float32))

    block = (threads_per_block, 1, 1)
    grid = (x.size // threads_per_block + 1, 1, 1)

    func_gpu_exp = mod.get_function("gpu_exp")
    func_gpu_exp(g_x, cuda.to_device(np.asarray(shape, np.int32)), grid=grid, block=block)

    func_gpu_sum = mod.get_function("gpu_sum")
    func_gpu_sum(g_x, g_x_sum, cuda.to_device(np.asarray(shape, np.int32)), grid=grid, block=block)

    func_gpu_div = mod.get_function("gpu_div")
    func_gpu_div(g_x, g_x_sum, cuda.to_device(np.asarray(shape, np.int32)), grid=grid, block=block)

    x = cuda.from_device(g_x, x.shape, x.dtype)

    return x

def gpu_relu_2d(x,threads_per_block=256):
    """
    :param x: [-1,10]
    :param threads_per_block:
    :return:
    """
    mod = SourceModule("""
      __global__ void gpu_relu(float *a_in,int* shape)
      {
        const int N=shape[0]*shape[1];
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
    g_x = cuda.to_device(x)
    block=(threads_per_block,1,1)
    grid = (x.size//threads_per_block+1,1,1)
    func(g_x,cuda.to_device(np.asarray(x.shape,np.int32)), grid=grid, block=block)
    x = cuda.from_device(g_x, x.shape, x.dtype)

    return x

def gpu_relu_4d(x):
    """
    :param x: [-1,24,24,20]
    :return:
    """
    mod = SourceModule(
        """
        __global__ void gpu_relu(float *a_inOut,int* shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x + bz*gridDim.x*gridDim.y;
            int offset = idx*blockDim.x+tx;
            a_inOut[offset]=a_inOut[offset]<0?0.0f:a_inOut[offset];
        }
        """
    )
    g_x = cuda.to_device(x)
    x_shape = x.shape
    g_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    block = (x_shape[-1],1,1)
    grid = (x_shape[2],x_shape[1],x_shape[0])
    func = mod.get_function("gpu_relu")
    func(g_x,g_shape,grid=grid, block=block)
    x = cuda.from_device(g_x, x.shape, x.dtype)
    return x

def gpu_BN(x,gamma,beta,mean,var,esp=1e-3):
    """
    :param x: [-1,24,24,20]
    :param gamma:[20,]
    :param beta:[20,]
    :param mean:[20,]
    :param var:[20,]
    :param esp:
    :return:
    """

    mod=SourceModule(
        """
        __global__ void gpu_bn(float *a_inOut,float *scale,float *shift)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x + bz*gridDim.x*gridDim.y;
            int offset = idx*blockDim.x+tx;
            a_inOut[offset]=a_inOut[offset]*scale[tx]+shift[tx];
        }
        """
    )

    scale = gamma / np.sqrt(var + esp)
    shift = -mean / np.sqrt(var + esp) * gamma + beta

    g_x = cuda.to_device(x)
    g_scale = cuda.to_device(scale)
    g_shift = cuda.to_device(shift)

    x_shape = x.shape
    # g_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    block = (x_shape[-1], 1, 1)
    grid = (x_shape[2], x_shape[1], x_shape[0])
    func = mod.get_function("gpu_bn")
    func(g_x,g_scale,g_shift, grid=grid, block=block)
    x = cuda.from_device(g_x, x.shape, x.dtype)
    return x

def conv2d_2(x,weight,bias,stream=0):
    """
    x:[-1,28,28,1] -->[-1,28,28,20] -- >[-1,24,24,20]
    w:[5,5,1,20] --> [20,5,5,1]
    b:[20]
    s:1
    p:valid
    """
    mod = DynamicSourceModule(
        """
        __global__ void gpu_conv2d(float *a_in,float *a_out,float *weight,float *bias,int *w_shape)
        {
            /*
            * a_in:[-1,28,28,1]
            * a_out:[-1,28,28,20]
            */
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tid = tx + ty * blockDim.x;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            // int idx = bid*blockDim.x*blockDim.y+tid;
            int new_idx =0;
            int w_idx =0;
            
            int cur_row=0,cur_col=0;
            int offset=0;
            float pixel=0.0f;
            
            
            for(int c=0;c<w_shape[0];++c)
            {
                float Csum=0.0f;
                for(int i=0;i<w_shape[1];++i)
                {
                    for(int j=0;j<w_shape[2];++j)
                    {
                        cur_row = by-w_shape[1]/2+i;
                        cur_col = bx-w_shape[2]/2+j; 
                        offset= (cur_col + cur_row*gridDim.x + bz*gridDim.x*gridDim.y)*blockDim.x*blockDim.y+tid;
                        if(cur_row<0 || cur_row>=gridDim.y || cur_col<0 || cur_col>= gridDim.x)
                            pixel = 0.0f;
                        else
                            pixel = a_in[offset];
                        
                        w_idx = c * w_shape[1]*w_shape[2]*w_shape[3]+i*w_shape[2]*w_shape[3]+j*w_shape[3]+tid;
                        // Csum += pixel*weight[((j+i*w_shape[1])*w_shape[2]+tid)*w_shape[-1]+c]; // 每个线程累加，不跨线程不会冲突
                        Csum += pixel*weight[w_idx]; // 每个线程累加，不跨线程不会冲突
                    }
                }
                new_idx=bid*w_shape[0]+c;
                atomicAdd(&a_out[new_idx],Csum); // 每个block跨线程相加，atomicAdd防止线程冲突
    
                // 加上bias
                if(tid==0)
                    a_out[new_idx]+=bias[c];
            }
        }
        
        // [-1,28,28,20] -- >[-1,24,24,20]
        __global__ void reduce_size(float *a_in,float *a_out,int *out_shape)
        {
            /*
            * a_in:[-1,28,28,20]
            * a_out:[-1,24,24,20]
            */
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tid = tx + ty * blockDim.x;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            int idx = bid*blockDim.x*blockDim.y+tid;

            // int weight_h=w_shape[1];
            // int weight_w=w_shape[2];
            
            if(bx<2 || by<2 || bx >=gridDim.x-2 || by>=gridDim.y-2) return;
            int new_bid=(bx-2) + (by-2)*out_shape[2] + bz*out_shape[2]*out_shape[1];
            int new_idx = new_bid*blockDim.x*blockDim.y+tid;

            a_out[new_idx] = a_in[idx];
        }
        
        """
    )

    x_shape = x.shape
    weight =np.transpose(weight,[3,0,1,2]).astype(np.float32) # [5,5,1,20] --> [20,5,5,1]
    w_shape = weight.shape
    out = np.zeros((*x_shape[:-1], w_shape[0]),np.float32)
    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)
    g_w = cuda.to_device(weight)
    g_b = cuda.to_device(bias)
    g_wshape = cuda.to_device(np.asarray([w_shape], np.int32))

    block = (x_shape[-1], 1, 1)
    grid = (x_shape[2], x_shape[1], x_shape[0])

    func = mod.get_function("gpu_conv2d")
    func(g_x, g_out, g_w, g_b, g_wshape, grid=grid, block=block, shared=0, stream=Stream(stream))

    # [-1,28,28,20] -- >[-1,24,24,20]
    out_shape =out.shape
    out2=np.zeros((out_shape[0],out_shape[1]+1-w_shape[1],out_shape[2]+1-w_shape[2],out_shape[3]),np.float32)
    g_out2=cuda.to_device(out2)
    g_out2shape = cuda.to_device(np.asarray([out2.shape], np.int32))

    block = (out_shape[-1], 1, 1)
    grid = (out_shape[2], out_shape[1], out_shape[0])
    func = mod.get_function("reduce_size")
    func(g_out, g_out2,g_out2shape,grid=grid, block=block, shared=0, stream=Stream(0))

    out2 = cuda.from_device(g_out2, out2.shape, out2.dtype)

    return out2
    # return out[:,2:x_shape[1]-2,2:x_shape[2]-2,:]

def conv2d(x, weight, bias):
    """
    使用多流处理
    x:[-1,28,28,1] -->[-1,28,28,20] -- >[-1,24,24,20]
    w:[5,5,1,20] --> [20,5,5,1]
    b:[20]
    s:1
    p:valid
    """

    # 将weight与bias拆开放到不同流中处理
    w_shape = weight.shape
    halfIndex=w_shape[0]//2+1
    out1=conv2d_2(x,weight[...,:halfIndex],bias[:halfIndex],0)
    out2=conv2d_2(x,weight[...,halfIndex:],bias[halfIndex:],1)

    # 合并结果
    out = np.concatenate((out1,out2),-1)
    return out

def maxpool(x):
    """
    x:[-1,24,24,20]
    out:[-1,24,24,20]
    s:2x2
    k:2x2
    """
    mod = SourceModule(
        """
        // #include <cmath>
        __global__ void maxpool2d(float *a_in,float *a_out,int* x_shape,int* w_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x + bz*gridDim.x*gridDim.y;
            int cur_row=0,cur_col=0;
            float pixel= -99999.0f;
            int offset=0;
            
            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;
                    offset = (cur_col + cur_row * gridDim.x + bz*gridDim.x*gridDim.y)*blockDim.x;
                    pixel=fmaxf(a_in[offset+tx],pixel);
                    // pixel = a_in[offset]>pixel?a_in[offset]:pixel;
                }
            }
            a_out[idx*blockDim.x+tx]=pixel;
        }
        
        // [-1,24,24,20] --> [-1,12,12,20]
        __global__ void lengthHalf(float *a_in,float *a_out)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x + bz*gridDim.x*gridDim.y;
            int offset=0;
            if(bx%2==0 || by%2==0) return;
            offset = bx/2 + by/2 * gridDim.x/2 + bz*gridDim.x/2*gridDim.y/2;// 减半
            a_out[offset*blockDim.x+tx] = a_in[idx*blockDim.x+tx];
        }
        
        """
    )

    x_shape = x.shape
    out1 = np.zeros([x_shape[0], x_shape[1], x_shape[2], x_shape[3]], x.dtype)
    out = np.zeros([x_shape[0], x_shape[1]//2, x_shape[2]//2, x_shape[3]], x.dtype)

    block = (x_shape[3], 1, 1)
    grid = (x_shape[2], x_shape[1], x_shape[0])

    gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    # gout_shape = cuda.to_device(np.asarray(out.shape, np.int32))
    gw_shape = cuda.to_device(np.asarray([2, 2], np.int32))

    g_x = cuda.to_device(x)
    g_out1 = cuda.to_device(out1)
    g_out = cuda.to_device(out)

    func = mod.get_function("maxpool2d")
    func(g_x, g_out1, gx_shape, gw_shape, grid=grid, block=block)


    func = mod.get_function("lengthHalf")
    func(g_out1, g_out, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def gpu_reshape(x):
    """
    :param x:[-1,24,24,20] -->[-1,24*24*20]
    :return:
    """

    mod = SourceModule(
        """
        __global__ void gpu_reshape(float *a_in,float *a_out,int *out_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x + bz*gridDim.x*gridDim.y;
            int offset = idx*blockDim.x+tx;
            a_out[offset] = a_in[offset];
        }
        """
    )

    x_shape = x.shape
    out = np.zeros([x_shape[0],x_shape[1]*x_shape[2]*x_shape[3]],np.float32)
    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)
    # g_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    gout_shape = cuda.to_device(np.asarray(out.shape, np.int32))
    block = (x_shape[-1], 1, 1)
    grid = (x_shape[2], x_shape[1], x_shape[0])
    func = mod.get_function("gpu_reshape")
    func(g_x,g_out,gout_shape, grid=grid, block=block)
    out = cuda.from_device(g_out, out.shape, out.dtype)
    return out

def gpu_matrix_mul_add_bias(x, wights, bias=None, threads_per_block=256):
    mod = SourceModule("""
          __global__ void gpu_matrix_mul(float *a_in,float *w,float *a_out,int *x_shape,int *w_shape)
          {
            // const int N_x = x_shape[0]*x_shape[1];
            // const int N_w = w_shape[0]*w_shape[1];
            const int x_row=x_shape[0];
            const int x_col=x_shape[1];

            const int w_row=w_shape[0];
            const int w_col=w_shape[1];

            int idx = threadIdx.x + blockIdx.x*blockDim.x;
        	// int i =idx/col; // 行索引
        	// int j =idx%col; // 列索引

            if (idx>=x_col || x_col!=w_row) return;

            //*
            // 每次处理一行与一列
            for(int i=0;i<x_row;++i)
            {
              for(int j=0;j<w_col;++j)
              {
                atomicAdd(&a_out[j+i*w_col],a_in[idx+i*x_col]*w[j+idx*w_col]);
              }
            }
            //*/

            /*
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
            */

          }

          __global__ void gpu_matrix_add_vector(float *a_inOut,float *bias,int *shape)
          {
            const int N = shape[0]*shape[1];
            // const int row=shape[0];
            const int col=shape[1];
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            // if(idx>=N) return;
            while(idx<N)
            {
                a_inOut[idx]+=bias[idx%col];
                idx+=blockDim.x * gridDim.x;
            }
          }

          """)

    func_matrix_mul = mod.get_function("gpu_matrix_mul")
    g_x = cuda.to_device(x)
    g_w = cuda.to_device(wights)
    x_shape = x.shape
    w_shape = wights.shape
    tmp = np.zeros([x_shape[0], w_shape[1]], np.float32)
    g_tmp = cuda.to_device(tmp)
    grid = (x_shape[1] // threads_per_block + 1, 1, 1)

    block = (threads_per_block, 1, 1)
    func_matrix_mul(g_x, g_w, g_tmp, cuda.to_device(np.asarray(x_shape, np.int32)),
                    cuda.to_device(np.asarray(w_shape, np.int32)), grid=grid, block=block)

    if bias is not None:
        g_b = cuda.to_device(bias)
        func_matrix_add_vector = mod.get_function("gpu_matrix_add_vector")
        grid = (tmp.size // threads_per_block + 1, 1, 1)
        # block = (threads_per_block, 1, 1)
        func_matrix_add_vector(g_tmp, g_b, cuda.to_device(np.asarray(tmp.shape, np.int32)), grid=grid, block=block)

    tmp = cuda.from_device(g_tmp, tmp.shape, tmp.dtype)

    return tmp

def model(parmas,img):
    """手动解析权重计算img的输出做分类"""
    x=np.reshape(img,[-1,28,28,1]).astype(np.float32)
    # x=conv2d_2(x,parmas["conv2d/kernel:0"],parmas["conv2d/bias:0"]) # 不使用分流
    x=conv2d(x,parmas["conv2d/kernel:0"],parmas["conv2d/bias:0"]) # 使用分流
    x = gpu_BN(x, parmas["batch_normalization/gamma:0"], parmas["batch_normalization/beta:0"],
    parmas["batch_normalization/moving_mean:0"], parmas["batch_normalization/moving_variance:0"])
    x=gpu_relu_4d(x)
    x=maxpool(x)

    x=conv2d(x,parmas["conv2d_1/kernel:0"],parmas["conv2d_1/bias:0"])
    x = gpu_BN(x, parmas["batch_normalization_1/gamma:0"], parmas["batch_normalization_1/beta:0"],
    parmas["batch_normalization_1/moving_mean:0"], parmas["batch_normalization_1/moving_variance:0"])
    x = gpu_relu_4d(x)
    x=maxpool(x)

    x=gpu_reshape(x)

    x=gpu_matrix_mul_add_bias(x,parmas['dense/kernel:0'],parmas['dense/bias:0'])
    x=gpu_relu_2d(x)
    x=gpu_matrix_mul_add_bias(x,parmas['dense_1/kernel:0'],parmas['dense_1/bias:0'])

    # softmax
    return gpu_softmax(x)

x_train, y_train, x_test, y_test=process_dataset()
start=time.time()
pred=model(parmas,x_test[:32])
print(time.time()-start)
print(np.argmax(pred,1))
print(y_test[:32])

"""
0.24666953086853027
[7 2 1 0 4 1 4 3 5 7 0 6 7 0 1 5 0 7 3 4]
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
"""
