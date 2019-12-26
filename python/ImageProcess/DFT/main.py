"""
离散傅里叶变换 DFT
"""
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
# from pycuda.compiler import DynamicSourceModule
from pycuda.autoinit import context
from pycuda.driver import Stream

# GPU
# 分块处理
def gpu_DCT(img, height, width):
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

        __global__ void dft2D(float *img,float *real,float *image,int *shape)
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
            float real_value = 0.0f;
            float image_value = 0.0f;
            for(int i =0;i<height/blockDim.y;++i)
            {
                for(int j=0;j<width/blockDim.x;++j)
                {
                    real_value += 1/(height*width)*img[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*cosf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height)); 
                    image_value += 1/(height*width)*img[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*sinf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height));             
                }
            }
            atomicAdd(&real[by*width+bx],real_value);
            atomicAdd(&image[by*width+bx],image_value);
        }
        """
    )

    func = mod.get_function("dft2D")
    g_img = cuda.to_device(img)
    real = np.zeros_like(img) # 实数部分
    image = np.zeros_like(img) # 虚数部分
    g_real = cuda.to_device(real)
    g_image = cuda.to_device(image)
    g_shape = cuda.to_device(np.asarray([height, width], np.int32))
    # block = (width, height, 1)
    block = (32, 32, 1)
    grid = (width, height, 1)
    func(g_img, g_real,g_image, g_shape, grid=grid, block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    real = cuda.from_device_like(g_real, real)
    image = cuda.from_device_like(g_image, image)

    return real,image,np.sqrt(real**2+image**2)

# DCT逆变换
def gpu_inv_DCT(real, image, height, width):
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

        __global__ void inv_dft2D(float *real,float *image,float *out_real,float *out_image,int *shape)
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
            float real_value = 0.0f;
            float image_value = 0.0f;
            for(int i =0;i<height/blockDim.y;++i)
            {
                for(int j=0;j<width/blockDim.x;++j)
                {
                    real_value += 1/(height*width)*(real[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*cosf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height))-
                    image[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*sinf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height)));
                    
                    image_value += 1/(height*width)*(image[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*cosf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height))+
                    real[(tx+j*blockDim.x)+(ty+i*blockDim.y)*width]*sinf(2*PI()*(bx*(tx+j*blockDim.x)/width+by*(ty+i*blockDim.y)/height)));                 
                }
            }
            atomicAdd(&out_real[by*width+bx],real_value);
            atomicAdd(&out_image[by*width+bx],image_value);
        }
        """
    )

    func = mod.get_function("inv_dft2D")
    g_real = cuda.to_device(real) # 实数
    g_image = cuda.to_device(image*(-1)) # 虚数
    out_real = np.zeros_like(real)
    out_image = np.zeros_like(image)
    g_out_real = cuda.to_device(out_real)
    g_out_image = cuda.to_device(out_image)
    g_shape = cuda.to_device(np.asarray([height, width], np.int32))
    # block = (width, height, 1)
    block = (32, 32, 1)
    grid = (width, height, 1)
    func(g_real, g_image,g_out_real,g_out_image, g_shape, grid=grid, block=block)
    # func(g_img,g_out,g_shape,grid=(),block=(),shared=0, stream=Stream(0))

    # GPU-->CPU
    out_real = cuda.from_device_like(g_out_real, out_real)
    out_image = cuda.from_device_like(g_out_image, out_image)
    return out_real,out_image,np.sqrt(real**2+image**2)

if __name__=="__main__":
    img = Image.open("test.jpg").convert("L").resize((160, 160))
    img = np.asarray(img, np.float32)
    height, width = img.shape
    real,image,out = gpu_DCT(img,height,width)
    # Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save("dct.jpg")
    Image.fromarray(np.clip(real,0,255).astype(np.uint8)).save("dct.jpg")

    # 逆变换
    out_real,out_image,out2 = gpu_inv_DCT(real,image,height,width)
    Image.fromarray(np.clip(out2, 0, 255).astype(np.uint8)).save("inv_dct.jpg")