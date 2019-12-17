"""
cuda 实现图片卷积
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
# import time
from PIL import Image
import sys

# 设置使用哪块GPU
cuda.Device(0)

def convelute(img:np.array,kernel:np.array)->np.array:
    """
    :param img: [300,500,3]
    :param kernel: [3,3,3]
    :return: [300,500]
    """
    mod = SourceModule(
        """
        __global__ void conv2d(float *img,float *kernel,float *out,int *wshape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            int tx = threadIdx.x;
            
            // int idx = tx + (bx+by*gridDim.x)*blockDim.x;
            int idx = bx+by*gridDim.x;
            
            int img_h = gridDim.y;
            int img_w = gridDim.x;
            // int img_c = blockDim.x;
            
            int cur_row=0,cur_col=0;
            int cur_idx = 0;
            float piexl = 0.0f;
            
            float value = 0.0f;
            for (int i=0;i<wshape[0];++i)
            {
                for (int j=0;j<wshape[1];++j)
                {
                    cur_row=by-wshape[0]/2+i;
                    cur_col=bx-wshape[1]/2+j;
                    cur_idx = tx + (cur_col+cur_row*gridDim.x)*blockDim.x;
                    
                    if (cur_row<0 || cur_col<0 || cur_row >=img_h || cur_col>=img_w)
                    {
                        piexl = 0.0f;
                    }
                    else
                        piexl = img[cur_idx];
                    
                    value += piexl* kernel[(j+i*wshape[1])*wshape[2]+tx];
                }
            }
            
            atomicAdd(&out[idx],value);
        }
        """
    )


    img_shape = img.shape
    kernel_shape = kernel.shape

    assert img_shape[-1]==kernel_shape[-1],"failure"

    g_img = cuda.to_device(img)
    g_kernel = cuda.to_device(kernel)
    # out = np.zeros_like(img)
    out = np.zeros([img_shape[0],img_shape[1]],np.float32)
    g_out = cuda.to_device(out)

    g_kernelShape = cuda.to_device(np.asarray(kernel_shape,np.int32))

    func = mod.get_function("conv2d")
    block = (img_shape[2],1,1)
    grid = (img_shape[1],img_shape[0],1)

    func(g_img,g_kernel,g_out,g_kernelShape,grid=grid,block=block)

    out = cuda.from_device(g_out,out.shape,out.dtype)

    return out


def convelute2(img: np.array, kernel: np.array) -> np.array:
    """
    :param img: [300,500,3]
    :param kernel: [3,3,3]
    :return: [300,500,3]
    """
    mod = SourceModule(
        """
        __global__ void conv2d(float *img,float *kernel,float *out,int *wshape)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;

            int tx = threadIdx.x;

            int idx = tx + (bx+by*gridDim.x)*blockDim.x;
            // int idx = bx+by*gridDim.x;

            int img_h = gridDim.y;
            int img_w = gridDim.x;
            // int img_c = blockDim.x;

            int cur_row=0,cur_col=0;
            int cur_idx = 0;
            float piexl = 0.0f;

            float value = 0.0f;
            for (int i=0;i<wshape[0];++i)
            {
                for (int j=0;j<wshape[1];++j)
                {
                    cur_row=by-wshape[0]/2+i;
                    cur_col=bx-wshape[1]/2+j;
                    cur_idx = tx + (cur_col+cur_row*gridDim.x)*blockDim.x;

                    if (cur_row<0 || cur_col<0 || cur_row >=img_h || cur_col>=img_w)
                    {
                        piexl = 0.0f;
                    }
                    else
                        piexl = img[cur_idx];

                    value += piexl* kernel[(j+i*wshape[1])*wshape[2]+tx];
                }
            }

            // atomicAdd(&out[idx],value);
            out[idx] = value;
        }
        """
    )

    img_shape = img.shape
    kernel_shape = kernel.shape

    assert img_shape[-1] == kernel_shape[-1], "failure"

    g_img = cuda.to_device(img)
    g_kernel = cuda.to_device(kernel)
    out = np.zeros_like(img)
    # out = np.zeros([img_shape[0], img_shape[1]], np.float32)
    g_out = cuda.to_device(out)

    g_kernelShape = cuda.to_device(np.asarray(kernel_shape, np.int32))

    func = mod.get_function("conv2d")
    block = (img_shape[2], 1, 1)
    grid = (img_shape[1], img_shape[0], 1)

    func(g_img, g_kernel, g_out, g_kernelShape, grid=grid, block=block)

    out = cuda.from_device(g_out, out.shape, out.dtype)

    return out

def main(argv):
    # assert len(argv)>1,print("参数不足， python3 convelution.py xxx/xx.jpg")
    img_path = "./image/test.jpg"
    if len(argv)>1:
        img_path = argv[1]

    img = Image.open(img_path).convert("RGB")
    img = np.asarray(img,np.float32)
    kernel = np.asarray([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]],np.float32)[...,None]
    # kernel = np.asarray([[-1, -1, -1],
    #                      [-1, 9, -1],
    #                      [-1, -1, -1]], np.float32)[..., None]
    # kernel = np.asarray([-1,0,1,-2,0,2,-1,0,1], np.float32).reshape(3,3)[..., None]
    # kernel = np.asarray([-1,-2,-1,0,0,0,1,2,1], np.float32).reshape(3,3)[..., None]
    kernel = np.concatenate((kernel,kernel,kernel),-1)

    # X方向—Sobel算子  -1,0,1,-2,0,2,-1,0,1
    # Y方向—Sobel算子  -1,-2,-1,0,0,0,1,2,1
    # Laplace算子      0, -1, 0, -1, 4, -1, 0, -1, 0

    # img = convelute(img,kernel)
    img = convelute2(img,kernel)

    img = np.clip(img,0.,255.).astype(np.uint8)
    Image.fromarray(img).save(img_path.replace(".jpg","_conv.jpg"))

if __name__=="__main__":
    main(sys.argv)