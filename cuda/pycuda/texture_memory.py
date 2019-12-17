import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
from PIL import Image


def texture1D():
    mod=SourceModule(
        """
        texture<float, 1 ,cudaReadModeElementType> tex; //声明1维 float类型的 纹理内存
        __global__ void copy(float *data)
        {
            // 从纹理内存中取出数据放入全局内存
            //int tx = threadIdx.x;
            //int idx = tx + blockIdx.x*blockDim.x;
            int idx = threadIdx.y;
            
            data[idx] = tex1D(tex,idx+0.5f);
        }
        
        """
    )

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 5)

    func = mod.get_function("copy")
    texref = mod.get_texref("tex")

    cuda.matrix_to_texref(a,texref,"C") # numpy array绑定到纹理内存
    # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    texref.addressMode = cuda.address_mode.WRAP
    texref.set_filter_mode(cuda.filter_mode.LINEAR)
    texref.normalized = True

    output = np.zeros_like(a)
    g_output = cuda.to_device(output)

    func(g_output,block=(1,5,1),grid=(1,1,1),shared=0,stream=Stream(0))

    output = cuda.from_device_like(g_output,output)

    return output

def texture2D():
    mod = SourceModule(
        """
        texture<float, 2 ,cudaReadModeElementType> tex; //声明2维 float类型的 纹理内存
        __global__ void copy(float *data)
        {
            // 从纹理内存中取出数据放入全局内存
            /*
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int idx = tx + ty * blockDim.x;
            data[idx] = tex2D(tex,tx+0.5f,ty+0.5f);
            */
            
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int idx = bx + by * gridDim.x;
            data[idx] = tex2D(tex,bx+0.5f,by+0.5f);
        }

        """
    )

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32).reshape(3, 3)

    func = mod.get_function("copy")
    texref = mod.get_texref("tex")

    cuda.matrix_to_texref(a, texref, "C")  # numpy array绑定到纹理内存
    # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    texref.addressMode = cuda.address_mode.WRAP
    texref.set_filter_mode(cuda.filter_mode.LINEAR)
    texref.normalized = True

    output = np.zeros_like(a)
    g_output = cuda.to_device(output)

    func(g_output, block=(1, 1, 1), grid=(3, 3, 1), shared=0, stream=Stream(0))

    output = cuda.from_device_like(g_output, output)

    return output

def imageRote():
    """
    实现图片旋转
    :return:
    """
    mod = SourceModule("""
        texture<float, 2 ,cudaReadModeElementType> tex; //声明2维 float类型的 纹理内存
        __global__ void image_rote(float *output,float *theta)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            int idx = bx + by * gridDim.x;
            
            // 按中心点旋转
            float u = (float)bx - (float)gridDim.x/2;
            float v = (float)by - (float)gridDim.y/2;
            float tu = u*cosf(theta[0]) - v*sinf(theta[0]); 
            float tv = v*cosf(theta[0]) + u*sinf(theta[0]); 
            
            // tu /= (float)gridDim.x; 
            // tv /= (float)gridDim.y;
            
            // read from texture and write to global memory
            // output[idx] = tex2D(tex, tu+0.5f, tv+0.5f);
            output[idx] = tex2D(tex, tu+gridDim.x/2.0f, tv+gridDim.y/2.0f);
            // output[idx] =  tex2D(tex,bx+0.5f,by+0.5f);  
        }

    """)

    img = Image.open("lena_bw.pgm").convert("L")
    img = np.asarray(img,np.float32)
    img_shape = img.shape
    output = np.zeros_like(img)
    g_output = cuda.to_device(output)

    texref = mod.get_texref("tex")
    cuda.matrix_to_texref(img,texref,"C")  # numpy array绑定到纹理内存
    texref.addressMode = cuda.address_mode.WRAP
    texref.set_filter_mode(cuda.filter_mode.LINEAR)
    texref.normalized = True

    g_theta = cuda.to_device(np.asarray([np.pi/8,],np.float32)) # 旋转弧度

    func = mod.get_function("image_rote")
    func(g_output,g_theta,
         block=(1, 1, 1), grid=(img_shape[1], img_shape[0], 1), shared=0, stream=Stream(0))

    context.synchronize() # Wait for kernel completion before host access

    output = cuda.from_device_like(g_output,output)
    output = np.clip(output,0.,255.)

    Image.fromarray(output.astype(np.uint8)).save("rote.pgm")



def main():
    # a = texture1D()
    # a = texture2D()
    # print(a)

    imageRote()

if __name__=="__main__":
    main()
