"""
离散余弦变换（DCT）：
https://blog.csdn.net/CHANG12358/article/details/82317894

DCT变换、DCT反变换、分块DCT变换
https://blog.csdn.net/weixin_30609331/article/details/98157347
"""
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
# from pycuda.compiler import DynamicSourceModule
from pycuda.autoinit import context
from pycuda.driver import Stream

def C(x):
    return 1/np.sqrt(2) if x==0 else 1

def cpu_DCT(img,out,height,width):
    for u in range(height): # 行
        for v in range(width): # 列
            for x in range(height):
                for y in range(width):
                    out[u,v] += img[x,y]*C(u)*C(v)*np.cos((2*x+1)*u*np.pi/(2*height))*np.cos((2*y+1)*v*np.pi/(2*width))

            out[u,v]*=2/(np.sqrt(height*width))

    return out

# GPU
def gpu_DCT(img,out,height,width):
    mod = SourceModule(
        """
        // #define PI 3.141592653589793
        
        __device__ float C(int x)
        {
            return x==0?1/sqrtf(2):1.0f;
        }
        
        __device__ float PI()
        {
            return acosf(-1.0f);
        }
        
        __global__ void dct2D(float *img,float *out,int *shape)
        {
            int height = shape[0];
            int width = shape[1];
            int tx = threadIdx.x; // y
            int ty = threadIdx.y; // x
            int bx = blockIdx.x; // v
            int by = blockIdx.y; // u
            
            atomicAdd(&out[by*width+bx],2/sqrtf(height*width)*
            img[tx+ty*width]*C(bx)*C(by)*cosf((2*ty+1)*by*PI()/(2*height))*
            cosf((2*tx+1)*bx*PI()/(2*width)));
        }
        """
    )

    func = mod.get_function("dct2D")
    g_img = cuda.to_device(img)
    g_out = cuda.to_device(out)
    g_shape = cuda.to_device(np.asarray([height,width],np.int32))
    block=(width,height,1)
    grid=(width,height,1)
    func(g_img,g_out,g_shape,grid=grid,block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    out = cuda.from_device_like(g_out,out)
    return out

# 分块处理
def gpu_DCT2(img, out, height, width):
    mod = SourceModule(
        """
        // #define PI 3.141592653589793

        __device__ float C(int x)
        {
            return x==0?1/sqrtf(2):1.0f;
        }

        __device__ float PI()
        {
            return acosf(-1.0f);
        }

        __global__ void dct2D(float *img,float *out,int *shape)
        {
            int height = shape[0];
            int width = shape[1];
            int tx = threadIdx.x; // y
            int ty = threadIdx.y; // x

            int bx = blockIdx.x; // v
            int by = blockIdx.y; // u
            
            // 0.每次计算一个块 blockDim.y * blockDim.x
            // 1.先每个block的每个线程各自相加
            // 2.再将每个block内部的线程结果相加 （减少同步次数，提高效率）
            float value = 0.0f;
            for(int i =0;i<height/blockDim.y;++i)
            {
                for(int j=0;j<width/blockDim.x;++j)
                {
                    value += 2/sqrtf(height*width)*C(bx)*C(by)*img[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*
                    cosf((2*(ty+i*blockDim.y)+1)*by*PI()/(2*height))*cosf((2*(tx+j*blockDim.x)+1)*bx*PI()/(2*width));                   
                }
            }
            atomicAdd(&out[by*width+bx],value);
        }
        """
    )

    func = mod.get_function("dct2D")
    g_img = cuda.to_device(img)
    g_out = cuda.to_device(out)
    g_shape = cuda.to_device(np.asarray([height, width], np.int32))
    # block = (width, height, 1)
    block = (32, 32, 1)
    grid = (width, height, 1)
    func(g_img, g_out, g_shape, grid=grid, block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    out = cuda.from_device_like(g_out, out)
    return out

# DCT逆变换
def gpu_inv_DCT(img,out,height,width):
    mod = SourceModule(
        """
        // #define PI 3.141592653589793

        __device__ float C(int x)
        {
            return x==0?1/sqrtf(2):1.0f;
        }

        __device__ float PI()
        {
            return acosf(-1.0f);
        }

        __global__ void inv_dct2D(float *img,float *out,int *shape)
        {
            int height = shape[0];
            int width = shape[1];
            int tx = threadIdx.x; // y
            int ty = threadIdx.y; // x
            int bx = blockIdx.x; // v
            int by = blockIdx.y; // u

            atomicAdd(&out[by*width+bx],2/sqrtf(height*width)*
            img[tx+ty*width]*C(tx)*C(ty)*cosf((2*by+1)*ty*PI()/(2*height))*
            cosf((2*bx+1)*tx*PI()/(2*width)));
        }
        """
    )

    func = mod.get_function("inv_dct2D")
    g_img = cuda.to_device(img)
    g_out = cuda.to_device(out)
    g_shape = cuda.to_device(np.asarray([height, width], np.int32))
    block = (width, height, 1)
    grid = (width, height, 1)
    func(g_img, g_out, g_shape, grid=grid, block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    out = cuda.from_device_like(g_out, out)
    return out


def gpu_inv_DCT2(img, out, height, width):
    mod = SourceModule(
        """
        // #define PI 3.141592653589793

        __device__ float C(int x)
        {
            return x==0?1/sqrtf(2):1.0f;
        }

        __device__ float PI()
        {
            return acosf(-1.0f);
        }

        __global__ void inv_dct2D(float *img,float *out,int *shape)
        {
            int height = shape[0];
            int width = shape[1];
            int tx = threadIdx.x; // y
            int ty = threadIdx.y; // x

            int bx = blockIdx.x; // v
            int by = blockIdx.y; // u

            // 0.每次计算一个块 blockDim.y * blockDim.x
            // 1.先每个block的每个线程各自相加
            // 2.再将每个block内部的线程结果相加 （减少同步次数，提高效率）
            float value = 0.0f;
            for(int i =0;i<height/blockDim.y;++i)
            {
                for(int j=0;j<width/blockDim.x;++j)
                {
                    value += 2/sqrtf(height*width)*C((tx+j*blockDim.x))*C((ty+i*blockDim.y))*img[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*
                    cosf((2*by+1)*(ty+i*blockDim.y)*PI()/(2*height))*cosf((2*bx+1)*(tx+j*blockDim.x)*PI()/(2*width));                   
                }
            }
            atomicAdd(&out[by*width+bx],value);
        }
        """
    )

    func = mod.get_function("inv_dct2D")
    g_img = cuda.to_device(img)
    g_out = cuda.to_device(out)
    g_shape = cuda.to_device(np.asarray([height, width], np.int32))
    # block = (width, height, 1)
    block = (32, 32, 1)
    grid = (width, height, 1)
    func(g_img, g_out, g_shape, grid=grid, block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    out = cuda.from_device_like(g_out, out)
    return out

if __name__=="__main__":
    img = Image.open("test.jpg").convert("L").resize((160, 160))
    img = np.asarray(img, np.float32)
    out = np.zeros_like(img)
    height, width = img.shape
    # out = gpu_DCT(img,out,height,width)
    out = gpu_DCT2(img,out,height,width)
    # print(out.shape)
    Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save("dct.jpg")

    # 逆变换
    out2 = np.zeros_like(img)
    # out2 = gpu_inv_DCT(out,out2,height,width)
    out2 = gpu_inv_DCT2(out,out2,height,width)
    Image.fromarray(np.clip(out2, 0, 255).astype(np.uint8)).save("inv_dct.jpg")