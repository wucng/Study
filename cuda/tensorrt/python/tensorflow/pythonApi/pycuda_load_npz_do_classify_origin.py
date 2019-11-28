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

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

def gpu_softmax(x, threads_per_block=256):
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

    shape = x.shape
    g_x = cuda.to_device(x)
    g_x_sum = cuda.to_device(np.zeros([x.shape[0], ], np.float32))

    block = (threads_per_block, 1, 1)
    grid = (x.size // threads_per_block + 1, 1, 1)

    func_gpu_exp = mod.get_function("gpu_exp")
    func_gpu_exp(g_x, cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    func_gpu_sum = mod.get_function("gpu_sum")
    func_gpu_sum(g_x, g_x_sum, cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    func_gpu_div = mod.get_function("gpu_div")
    func_gpu_div(g_x, g_x_sum, cuda.to_device(np.asarray(shape, np.float32)), grid=grid, block=block)

    x = cuda.from_device(g_x, x.shape, x.dtype)

    return x

def relu(x):
    s = np.where(x < 0, 0, x)
    return s

def gpu_relu_2d(x,threads_per_block=256):
    """
    :param x: [-1,10]
    :param threads_per_block:
    :return:
    """
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
    g_x = cuda.to_device(x)
    block=(threads_per_block,1,1)
    grid = (x.size//threads_per_block+1,1,1)
    func(g_x,cuda.to_device(np.asarray(x.shape,np.float32)), grid=grid, block=block)
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


def batch_norm(x,gamma,beta,mean,var,esp=1e-3):
    scale = gamma/np.sqrt(var+esp)
    shift = -mean/np.sqrt(var+esp)*gamma+beta

    return x*scale+shift

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
    g_scale = cuda.to_device(scale.astype(np.float32))
    g_shift = cuda.to_device(shift.astype(np.float32))

    x_shape = x.shape
    # g_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    block = (x_shape[-1], 1, 1)
    grid = (x_shape[2], x_shape[1], x_shape[0])
    func = mod.get_function("gpu_bn")
    func(g_x,g_scale,g_shift, grid=grid, block=block)
    x = cuda.from_device(g_x, x.shape, x.dtype)
    return x


def cpu_conv2d(x,out,weight,bias,x_shape,w_shape):
    """
    x:[28,28,1]
    w:[5,5,1]
    b:[1]
    return : [28,28,1]
    """
    # pixel=0.0
    for m in range(x_shape[0]):
        for n in range(x_shape[1]):
            for i in range(w_shape[0]):
                for j in range(w_shape[1]):
                    cur_row = m-w_shape[0]//2+i
                    cur_col = n-w_shape[1]//2+j
                    for k in range(w_shape[2]):
                        if cur_row<0 or cur_row>=x_shape[0] or cur_col<0 or cur_col>=x_shape[1]:
                            pixel=0
                        else:
                            pixel=x[cur_row,cur_col,k]

                        out[m,n]+=pixel*weight[i,j,k]

            out[m,n]+=bias

    return out

def gpu_conv2d_2(x, out, weight, bias, x_shape, w_shape):
    """
    x:[28,28,1]
    w:[5,5,1]
    b:[1]
    return : [28,28,1]
    """
    mod = SourceModule(
        """
        __global__ void conv2d(float *a_in,float *a_out,float *weight,float *bias,int* x_shape,int* w_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;

            int idx = bx + by * gridDim.x;

            int cur_row=0,cur_col=0;
            int offset=0;
            float pixel=0.0f;

            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;                    
                    offset=cur_row*gridDim.x+cur_col;

                    if(cur_row<0 || cur_row>=x_shape[0] || cur_col<0 || cur_col>= x_shape[1])
                        pixel = 0.0f;
                    else
                        pixel = a_in[offset*w_shape[2]+tx];

                    //a_out[idx] += pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx];// 每个会有blockDim.x个线程同时执行，这种会有冲突
                    atomicAdd(&a_out[idx],pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx]);
                }
            }

            // 加上bias
            if(tx==0)
                a_out[idx]+=bias[0];
        }
        """
    )

    block = (x_shape[-1], 1, 1)
    # grid = (x_shape[0], x_shape[1], 1)
    grid = (x_shape[1], x_shape[0], 1)  # grid: x,y,z ; x shape: n,h,w,c

    gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    gw_shape = cuda.to_device(np.asarray(w_shape, np.int32))

    g_x = cuda.to_device(x)
    g_w = cuda.to_device(weight)
    g_b = cuda.to_device(np.asarray([bias], np.float32))
    g_out = cuda.to_device(out)

    func = mod.get_function("conv2d")
    func(g_x, g_out, g_w, g_b, gx_shape, gw_shape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def gpu_conv2d_3(x, out, weight, bias, x_shape, w_shape):
    """
    x:[28,28,1]
    w:[5,5,1]
    b:[1]
    return : [28,28,1]
    """
    mod = SourceModule(
        """
        __global__ void conv2d(float *a_in,float *a_out,float *weight,float *bias,int* x_shape,int* w_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;

            int idx = bx + by * gridDim.x;

            int cur_row=0,cur_col=0;
            int offset=0;
            float pixel=0.0f;
            float Csum=0.0f; // block每个线程都有一个局部变量 

            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;                    
                    offset=cur_row*gridDim.x+cur_col;

                    if(cur_row<0 || cur_row>=x_shape[0] || cur_col<0 || cur_col>= x_shape[1])
                        pixel = 0.0f;
                    else
                        pixel = a_in[offset*w_shape[2]+tx];

                    // a_out[idx] += pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx];// 每个会有blockDim.x个线程同时执行，这种会有冲突
                    // atomicAdd(&a_out[idx],pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx]);// 可以拆成以下两步，1.线程内加 2.线程间加

                    Csum += pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx];// 先让每个线程内部进行相加，不跨线程不会出现冲突
                }
            }

            // 将每个block内所有线程计算结果相加（涉及到多个线程之间操作，会有冲突，这时使用原子加解决冲突，但使用atomicAdd会造成线程同步等待而浪费时间）
            atomicAdd(&a_out[idx],Csum);

            // 加上bias
            if(tx==0)
                a_out[idx]+=bias[0];
        }
        """
    )

    block = (x_shape[-1], 1, 1)
    # grid = (x_shape[0], x_shape[1], 1)
    grid = (x_shape[1], x_shape[0], 1)  # grid: x,y,z ; x shape: n,h,w,c

    gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    gw_shape = cuda.to_device(np.asarray(w_shape, np.int32))

    g_x = cuda.to_device(x)
    g_w = cuda.to_device(weight)
    g_b = cuda.to_device(np.asarray([bias], np.float32))
    g_out = cuda.to_device(out)

    func = mod.get_function("conv2d")
    func(g_x, g_out, g_w, g_b, gx_shape, gw_shape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def gpu_conv2d(x,out,weight,bias,x_shape,w_shape):
    """
    x:[28,28,1]
    w:[5,5,1]
    b:[1]
    return : [28,28,1]
    """
    mod = SourceModule(
        """
        __global__ void conv2d(float *a_in,float *a_out,float *weight,float *bias,int* x_shape,int* w_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            // int tx = threadIdx.x;
            
            int idx = bx + by * gridDim.x;
            
            int cur_row=0,cur_col=0;
            int offset=0;
            float pixel=0.0f;
            
            int img_h=(int)x_shape[0];
            int img_w=(int)x_shape[1];
            int kernel_h=(int)w_shape[0];
            int kernel_w=(int)w_shape[1];
            int kernel_c=(int)w_shape[2];
            
            for(int i=0;i<kernel_h;++i)
            {
                for(int j=0;j<kernel_w;++j)
                {
                    cur_row = by-kernel_h/2+i;
                    cur_col = bx-kernel_w/2+j;
                    
                    offset=cur_row*gridDim.x+cur_col;
                    
                    for(int k=0;k<kernel_c;++k)
                    {
                        if(cur_row<0 || cur_row>=img_h || cur_col<0 || cur_col>= img_w)
                            pixel = 0.0f;
                        else
                            pixel = a_in[offset*kernel_c+k];
                        
                        a_out[idx] += pixel*weight[(j+i*kernel_w)*kernel_c+k];
                    }                   

                }
            }
            
            // 加上bias
            a_out[idx]+=bias[0];
        }
        """
    )

    block = (1, 1, 1)
    # grid = (x_shape[0], x_shape[1], 1)
    grid = (x_shape[1], x_shape[0], 1) # grid: x,y,z ; x shape: n,h,w,c

    gx_shape=cuda.to_device(np.asarray(x_shape,np.int32))
    gw_shape = cuda.to_device(np.asarray(w_shape, np.int32))

    g_x=cuda.to_device(x)
    g_w=cuda.to_device(weight)
    g_b=cuda.to_device(np.asarray([bias],np.float32))
    g_out=cuda.to_device(out)

    func = mod.get_function("conv2d")
    func(g_x,g_out,g_w,g_b,gx_shape,gw_shape,grid=grid,block=block)

    out=cuda.from_device(g_out,out.shape,out.dtype)

    return out

def conv2d2(x,weight,bias):
    """
    x:[-1,28,28,1] -->[-1,24,24,20]
    w:[5,5,1,20]
    b:[20,]
    s:1
    p:valid
    """
    x = x.astype(np.float32)

    x_shape = x.shape
    w_shape = weight.shape
    out = np.zeros((*x_shape[:-1],w_shape[-1])).astype(np.float32)
    for i in range(x_shape[0]):
        img=x[i,...]
        for j in range(w_shape[-1]):
            # out[i,:,:,j]=cpu_conv2d(img,out[i,:,:,j],weight[...,j],bias[j],x_shape[1:],w_shape[:-1])
            out[i,:,:,j]=gpu_conv2d_3(img,out[i,:,:,j],weight[...,j],bias[j],x_shape[1:],w_shape[:-1])
    # [-1,28,28,1] -->[-1,24,24,20]
    return out[:,2:x_shape[1]-2,2:x_shape[2]-2,:]

def gpu_conv2d_4(x,out,weight,bias,x_shape,w_shape,stream=0):
    """
    x:[-1,28,28,1] -->[-1,28,28,1]
    w:[5,5,1]
    b:[1,]
    s:1
    p:valid
    """
    mod = DynamicSourceModule(
        """
        __global__ void gpu_conv2d(float *a_in,float *a_out,float *weight,float *bias,int *w_shape)
        {
            /*
            * a_in:[-1,24,24,3]
            * a_out:[-1,24,24,1]
            */
            
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int bz = blockIdx.z;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tid = tx + ty * blockDim.x;
            int bid = bx + by*gridDim.x + bz*gridDim.x*gridDim.y;
            // int idx = bid*blockDim.x*blockDim.y+tid;
            
            
            int cur_row=0,cur_col=0;
            int offset=0;
            float pixel=0.0f;
            float Csum=0.0f;
            
            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j; 
                    offset= (cur_col + cur_row*gridDim.x + bz*gridDim.x*gridDim.y)*blockDim.x+tid;
                    if(cur_row<0 || cur_row>=gridDim.y || cur_col<0 || cur_col>= gridDim.x)
                        pixel = 0.0f;
                    else
                        pixel = a_in[offset];
                    
                    Csum += pixel*weight[(j+i*w_shape[1])*w_shape[2]+tid]; // 每个线程累加，不跨线程不会冲突
                }
            }
            atomicAdd(&a_out[bid],Csum); // 每个block跨线程相加，atomicAdd防止线程冲突
            
            // 加上bias
            if(tid==0)
                a_out[bid]+=bias[0];
        }
        """
    )

    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)
    g_w = cuda.to_device(weight)
    g_b = cuda.to_device(np.asarray([bias], np.float32))
    g_wshape=cuda.to_device(np.asarray([w_shape],np.int32))

    block=(x_shape[-1],1,1)
    grid=(x_shape[2],x_shape[1],x_shape[0])

    func = mod.get_function("gpu_conv2d")
    func(g_x,g_out,g_w,g_b,g_wshape,grid=grid,block=block,shared=0,stream=Stream(stream))

    out=cuda.from_device(g_out,out.shape,out.dtype)

    return out

def conv2d1(x,weight,bias):
    """
    x:[-1,28,28,1] -->[-1,24,24,20]
    w:[5,5,1,20]
    b:[20,]
    s:1
    p:valid
    """
    x = x.astype(np.float32)

    x_shape = x.shape
    w_shape = weight.shape
    out = np.zeros((*x_shape[:-1],w_shape[-1])).astype(np.float32)

    for j in range(w_shape[-1]):
        out[...,j]=gpu_conv2d_4(x,out[...,j],weight[...,j],bias[j],x_shape,w_shape,0)

    # [-1,28,28,1] -->[-1,24,24,20]
    return out[:,2:x_shape[1]-2,2:x_shape[2]-2,:]

def conv2d(x,weight,bias):
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

    x = x.astype(np.float32)
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
    func(g_x, g_out, g_w, g_b, g_wshape, grid=grid, block=block, shared=0, stream=Stream(0))

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

def cpu_maxpool(x,out,x_shape):
    """
    x:[24,24]
    out:[12,12]
    s:2x2
    k:2x2
    """
    kernel_h=kernel_w=2
    for m in range(x_shape[0]):
        for n in range(x_shape[1]):
            if (m+n)%2==0 and m+n>0:
                for i in range(kernel_h): # kernel_h
                    for j in range(kernel_w): # kernel_w
                        cur_row = m-kernel_h//2+i
                        cur_col = n-kernel_w//2+j
                        pixel=x[cur_row,cur_col]
                        # 取最大
                        out[m//2,n//2]=max(out[m//2,n//2],pixel)
    
    return out

def gpu_maxpool2(x,out,x_shape):
    """
    x:[24,24]
    out:[12,12]
    s:2x2
    k:2x2
    """
    mod=SourceModule(
        """
        // #include <cmath>
        __global__ void maxpool2d(float *a_in,float *a_out,int* out_shape,int* w_shape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            // int idx = bx + by * gridDim.x;
            int cur_row=0,cur_col=0;
            float pixel= -99999.0f;
            int offset=0;
            
            // if((bx+by)>0 && (bx+by)%2==0)
            if((bx+by)<=0 || (bx+by)%2!=0) return;
            
            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;
                    offset = cur_col + cur_row * gridDim.x;
                    // pixel=a_in[offset];
                    // a_out[by/2,bx/2]=fmaxf(a_out[by/2,bx/2],pixel);// 每个block只有一个线程不会冲突
                    // a_out[out_shape[1]*by/2+bx/2]=fmaxf(a_out[out_shape[1]*by/2+bx/2],pixel);
                    
                    pixel=fmaxf(a_in[offset],pixel);
                }
            }
            a_out[out_shape[1]*by/2+bx/2]=pixel;
        }
        """
    )

    block = (1, 1, 1)
    grid = (x_shape[1], x_shape[0], 1)

    # gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    gout_shape = cuda.to_device(np.asarray(out.shape, np.int32))
    gw_shape = cuda.to_device(np.asarray([2,2], np.int32))

    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)

    func = mod.get_function("maxpool2d")
    func(g_x, g_out, gout_shape, gw_shape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def maxpool2(x):
    """
    x:[-1,24,24,20]-->[-1,12,12,20]
    s:2x2
    k:2x2
    """
    x = x.astype(np.float32)
    x_shape = x.shape
    out = np.zeros([x_shape[0],x_shape[1]//2,x_shape[2]//2,x_shape[3]],x.dtype)
    for i in range(x_shape[0]):
        for j in range(x_shape[-1]):
            # out[i,:,:,j]=cpu_maxpool(x[i,:,:,j],out[i,:,:,j],(x_shape[1],x_shape[2]))
            out[i,:,:,j]=gpu_maxpool(x[i,:,:,j],out[i,:,:,j],(x_shape[1],x_shape[2]))

    return out

def gpu_maxpool1(x, out, x_shape):
    """
    x:[24,24]
    out:[24,24]
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
            int idx = bx + by * gridDim.x;
            int cur_row=0,cur_col=0;
            float pixel= -99999.0f;
            int offset=0;

            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;
                    offset = cur_col + cur_row * gridDim.x;
                    pixel=fmaxf(a_in[offset],pixel);
                    // pixel = a_in[offset]>pixel?a_in[offset]:pixel;
                }
            }
            a_out[idx]=pixel;
        }
        """
    )

    block = (1, 1, 1)
    grid = (x_shape[1], x_shape[0], 1)

    gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    # gout_shape = cuda.to_device(np.asarray(out.shape, np.int32))
    gw_shape = cuda.to_device(np.asarray([2, 2], np.int32))

    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)

    func = mod.get_function("maxpool2d")
    func(g_x, g_out, gx_shape, gw_shape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def maxpool1(x):
    """
    x:[-1,24,24,20]-->[-1,12,12,20]
    s:2x2
    k:2x2
    """
    x = x.astype(np.float32)
    x_shape = x.shape
    out = np.zeros([x_shape[0], x_shape[1], x_shape[2], x_shape[3]], x.dtype)
    start = time.time()
    for i in range(x_shape[0]):
        for j in range(x_shape[-1]):
            # out[i,:,:,j]=cpu_maxpool(x[i,:,:,j],out[i,:,:,j],(x_shape[1],x_shape[2]))
            out[i, :, :, j] = gpu_maxpool(x[i, :, :, j], out[i, :, :, j], (x_shape[1], x_shape[2]))
    print(time.time()-start)
    exit(0)

    # [-1,24,24,20]-->[-1,12,12,20]

    return out[:,1:x_shape[1]:2,1:x_shape[2]:2,:]

def gpu_maxpool0(x, out, x_shape):
    """
    x:[24,24,20]
    out:[24,24,20]
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
            int tx = threadIdx.x;
            int idx = bx + by * gridDim.x;
            int cur_row=0,cur_col=0;
            float pixel= -99999.0f;
            int offset=0;

            for(int i=0;i<w_shape[0];++i)
            {
                for(int j=0;j<w_shape[1];++j)
                {
                    cur_row = by-w_shape[0]/2+i;
                    cur_col = bx-w_shape[1]/2+j;
                    offset = (cur_col + cur_row * gridDim.x)*blockDim.x;
                    pixel=fmaxf(a_in[offset+tx],pixel);
                    // pixel = a_in[offset]>pixel?a_in[offset]:pixel;
                }
            }
            a_out[idx*blockDim.x+tx]=pixel;
        }
        """
    )

    block = (x_shape[2], 1, 1)
    grid = (x_shape[1], x_shape[0], 1)

    gx_shape = cuda.to_device(np.asarray(x_shape, np.int32))
    # gout_shape = cuda.to_device(np.asarray(out.shape, np.int32))
    gw_shape = cuda.to_device(np.asarray([2, 2], np.int32))

    g_x = cuda.to_device(x)
    g_out = cuda.to_device(out)

    func = mod.get_function("maxpool2d")
    func(g_x, g_out, gx_shape, gw_shape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def maxpool0(x):
    """
    x:[-1,24,24,20]-->[-1,12,12,20]
    s:2x2
    k:2x2
    """
    x = x.astype(np.float32)
    x_shape = x.shape
    out = np.zeros([x_shape[0], x_shape[1], x_shape[2], x_shape[3]], x.dtype)
    start = time.time()
    for i in range(x_shape[0]):
        out[i,...] = gpu_maxpool(x[i,...], out[i,...], x_shape[1:])
    print(time.time()-start)
    # exit(0)

    # [-1,24,24,20]-->[-1,12,12,20]
    return out[:,1:x_shape[1]:2,1:x_shape[2]:2,:]

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

    x = x.astype(np.float32)
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
    x=x.astype(np.float32)
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

    func_matrix_mul = mod.get_function("gpu_matrix_mul")
    g_x = cuda.to_device(x)
    g_w = cuda.to_device(wights)
    x_shape = x.shape
    w_shape = wights.shape
    tmp = np.zeros([x_shape[0], w_shape[1]], np.float32)
    g_tmp = cuda.to_device(tmp)
    grid = (x_shape[1] // threads_per_block + 1, 1, 1)

    block = (threads_per_block, 1, 1)
    func_matrix_mul(g_x, g_w, g_tmp, cuda.to_device(np.asarray(x_shape, np.float32)),
                    cuda.to_device(np.asarray(w_shape, np.float32)), grid=grid, block=block)

    if bias is not None:
        g_b = cuda.to_device(bias)
        func_matrix_add_vector = mod.get_function("gpu_matrix_add_vector")
        grid = (tmp.size // threads_per_block + 1, 1, 1)
        # block = (threads_per_block, 1, 1)
        func_matrix_add_vector(g_tmp, g_b, cuda.to_device(np.asarray(tmp.shape, np.float32)), grid=grid, block=block)

    tmp = cuda.from_device(g_tmp, tmp.shape, tmp.dtype)

    return tmp

def model(parmas,img):
    """手动解析权重计算img的输出做分类"""
    x=np.reshape(img,[-1,28,28,1]).astype(np.float32)
    x=conv2d(x,parmas["conv2d/kernel:0"],parmas["conv2d/bias:0"])
    # x=batch_norm(x,parmas["batch_normalization/gamma:0"],parmas["batch_normalization/beta:0"],
    # parmas["batch_normalization/moving_mean:0"],parmas["batch_normalization/moving_variance:0"])
    x = gpu_BN(x, parmas["batch_normalization/gamma:0"], parmas["batch_normalization/beta:0"],
       parmas["batch_normalization/moving_mean:0"], parmas["batch_normalization/moving_variance:0"])
    # x=relu(x)
    x=gpu_relu_4d(x)
    x=maxpool(x)

    x=conv2d(x,parmas["conv2d_1/kernel:0"],parmas["conv2d_1/bias:0"])
    # x=conv2d2(x,parmas["conv2d_1/kernel:0"],parmas["conv2d_1/bias:0"])

    # x=batch_norm(x,parmas["batch_normalization_1/gamma:0"],parmas["batch_normalization_1/beta:0"],
    # parmas["batch_normalization_1/moving_mean:0"],parmas["batch_normalization_1/moving_variance:0"])
    x = gpu_BN(x, parmas["batch_normalization_1/gamma:0"], parmas["batch_normalization_1/beta:0"],
           parmas["batch_normalization_1/moving_mean:0"], parmas["batch_normalization_1/moving_variance:0"])
    # x=relu(x)
    x = gpu_relu_4d(x)
    x=maxpool(x)

    # x = np.reshape(x,[x.shape[0],-1])
    x=gpu_reshape(x)

    # x=np.matmul(x,parmas['dense/kernel:0'])+parmas['dense/bias:0']
    x=gpu_matrix_mul_add_bias(x,parmas['dense/kernel:0'],parmas['dense/bias:0'])
    # x=relu(x)
    x=gpu_relu_2d(x)
    # x=np.matmul(x,parmas['dense_1/kernel:0'])+parmas['dense_1/bias:0']
    x=gpu_matrix_mul_add_bias(x,parmas['dense_1/kernel:0'],parmas['dense_1/bias:0'])

    # softmax
    # return softmax(x)
    return gpu_softmax(x)
    # return x

start=time.time()
x_train, y_train, x_test, y_test=process_dataset()
print(np.argmax(model(parmas,x_test[:20]),1))
print(y_test[:20])
print(time.time()-start)
"""
[7 2 1 0 4 1 4 3 5 7 0 6 7 0 1 5 0 7 3 4]
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
0.7152385711669922
"""
