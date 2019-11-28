"""
测试gpu实现卷积操作
"""
import numpy as np
import cv2
import PIL.Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

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

def gpu_conv2d(x, out, weight, bias, x_shape, w_shape):
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

                    // a_out[idx] += pixel*weight[(j+i*w_shape[1])*w_shape[2]+tx];// 每个会有blockDim.x个线程同时执行，这种会有冲突
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


'''
flag="RGB"
# flag="L"

# 打开图片
img=PIL.Image.open("test.jpg")
weight=[[0,-1,0],
        [-1,5,-1],
        [0,-1,0]]
weight=np.asarray(weight,np.float32)[...,None]
if flag=="RGB":
    img=img.convert('RGB')
    img = np.asarray(img, np.float32)
    weight=np.concatenate((weight,weight,weight),-1)
else:
    img=img.convert('L')
    # img=img.resize((224, 224))
    img = np.asarray(img,np.float32)
    img=img[...,None]

x_shape=img.shape
out=np.zeros(x_shape[:2],np.float32)

bias=np.asarray([0],np.float32)

# out=gpu_conv2d(img,out,weight,bias,x_shape,weight.shape)
out=gpu_conv2d_2(img,out,weight,bias,x_shape,weight.shape)

out=np.clip(out,0,255).astype(np.uint8)

PIL.Image.fromarray(out).save("new2.jpg",'jpeg')
'''

x=np.ones([5,5],np.float32)[...,None]
out=np.zeros(x.shape[:2],np.float32)
weight=[[0,-1,0],
        [-1,5,-1],
        [0,-1,0]]
weight=np.asarray(weight,np.float32)[...,None]
bias=np.asarray([0],np.float32)

start=time.time()
out1 = gpu_conv2d(x,out,weight,bias,x.shape,weight.shape)
print(time.time()-start)

start=time.time()
out2 = gpu_conv2d_2(x,out,weight,bias,x.shape,weight.shape)
print(time.time()-start)

start=time.time()
out3 = gpu_conv2d_3(x,out,weight,bias,x.shape,weight.shape)
print(time.time()-start)

start=time.time()
out4 = cpu_conv2d(x,out,weight,bias,x.shape,weight.shape)
print(time.time()-start)

# """
print("-------------out1-------------")
print(out1)

print("-------------out2-------------")
print(out2)

print("-------------out3-------------")
print(out3)

print("-------------out4-------------")
print(out4)
# """

print("-------------error-------------")
print(np.max(np.fabs(out1-out2)))