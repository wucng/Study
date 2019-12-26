<font size=4>

- [离散余弦变换（DCT）](https://blog.csdn.net/CHANG12358/article/details/82317894)
- [zigzag模式提取矩阵元素](https://blog.csdn.net/zouxy09/article/details/13298817)
- [DCT变换、DCT反变换、分块DCT变换](https://blog.csdn.net/weixin_30609331/article/details/98157347)

---

[toc]

# zigzag
![](https://img-blog.csdn.net/20131028191537703?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvem91eHkwOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

用在离散余弦变换的系数提取里面。离散余弦变换（DCT）是种图像压缩算法，JPEG-2000好像就是用它来进行图像压缩的。DCT将图像从像素域变换到频率域。然后一般图像都存在很多冗余和相关性的，所以转换到频率域之后，只有很少的一部分频率分量的系数才不为0，大部分系数都为0（或者说接近于0），这样就可以进行高效的编码，以达到压缩的目的。下图的右图是对lena图进行离散余弦变换（DCT）得到的系数矩阵图。从左上角依次到右下角，频率越来越高，由图可以看到，左上角的值比较大，到右下角的值就很小很小了。换句话说，图像的能量几乎都集中在左上角这个地方的低频系数上面了。

![](https://img-blog.csdn.net/20131028191551453?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvem91eHkwOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
而且，系数的重要性一定程度上是按照zigzag模式进行排列的。所以就可以通过zigzag模式来提取这个矩阵前面的重要性的元素，作为这个图像在`频率域上的特征`，然后可以拿去做分类啥的，以达到降维的功效。`(使用zigzag模式 提取经过DCT变换后的频率特征，作为这张图像的特征)`

```python
"""
初始矩阵需满足 m x n  (m=n)
如果不满足可以填充成 行与列相同，在得到的结果中把填充值去掉就得到最终结果
"""

import numpy as np

mat = np.asarray([[1,2,6,3],[3,5,7,8],[0,0,0,0],[0,0,0,0]],np.uint8)
h,w = mat.shape

# out = np.zeros(mat.size,np.uint8) # 输出
out =[]

# 1.水平方向矩阵 mat[0,:] ,水平索引值 [0,w-1]
h_mat = mat[0,:]
h_i =0
h_end =w-1

# 2.对角方向矩阵 mat[h:0:-1,1:]
d_mat = mat[h:0:-1,1:]
h_d,w_d = d_mat.shape
if h_d>w_d:
    d_i = w_d//2-h_d
    d_end = w_d//2
else:
    d_i = -h_d//2
    d_end = w_d+d_i

# 3.垂直方向矩阵 mat[1:,0],垂直索引 [0,h-2]
v_mat = mat[1:,0]
v_i = 0
v_end = h-2

# 实施
# 水平方向和垂直方向每次取2个元素，只有1个取1个，没有则跳过
# 对角方向每次取一个对角元素，偶数时需反转
flag = True
while True:
    # 1.水平方向
    if h_i <=h_end:
        out.extend(h_mat[h_i:h_i+2])
        h_i+=2

    # 2.对角方向（向下,需反转） (第一次时不写入，可以认为存在一个空对角元素)
    if d_i >d_end:
        break
    if not flag:
        out.extend(list(reversed(d_mat.diagonal(d_i))))
        d_i+=1

    # 3.垂直方向
    if v_i <=v_end:
        out.extend(v_mat[v_i:v_i+2])
        v_i+=2

    # 4.对角（向上）
    out.extend(d_mat.diagonal(d_i))
    d_i += 1

    flag = False

print(mat)
print(out)
# [1, 2, 3, 0, 5, 6, 3, 7, 0, 0, 0, 0, 8, 0, 0, 0]
# [value for value in out if value != 0]
# 去掉填充值0 得到最后结果：[1, 2, 3, 5, 6, 3, 7, 8]
```

# 结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220164719347.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
<center>原图</center>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220164726138.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
<center>DCT变换到频率域图</center>


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220164736244.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
<center>DCT反变换到空间域图</center>

# 二维DCT
一维离散余弦变换定义:
![](https://img-blog.csdn.net/20161029163654430)
一维DCT定义如下： 设{f(x)|x=0,  1,  …,  N-1}为离散的信号列

![](https://img-blog.csdn.net/20161029163922120)

---
二维DCT定义如下：

设f(x,  y)为M×N的数字图像矩阵，则
![](https://img-blog.csdn.net/20161029164539691)

式中： x,  u=0,  1,  2,  …,  M－1； y,  v=0,  1,  2,  …,  N－1。

通常根据可分离性， 二维DCT可用两次一维DCT来完成， 其算法流程与DFT类似， 即
![](https://img-blog.csdn.net/20161029164710381)

---
or

![](https://images0.cnblogs.com/blog/661248/201408/311243394542578.png)

```python
"""
离散余弦变换（DCT）：
https://blog.csdn.net/CHANG12358/article/details/82317894

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
```

---
```python
File "/usr/lib/python3/dist-packages/pycuda/compiler.py", line 137, in compile_plain
    stderr=stderr.decode("utf-8", "replace"))
pycuda.driver.CompileError: nvcc compilation of /tmp/tmpse323ibf/kernel.cu failed
[command: nvcc --cubin -arch sm_75 -I/usr/lib/python3/dist-packages/pycuda/cuda kernel.cu]
[stderr:
nvcc fatal   : Value 'sm_75' is not defined for option 'gpu-architecture'
]

# 解决方法：
# 1.手动编译 只会生成 kernel.cubin
nvcc --cubin -arch sm_70 -I/usr/lib/python3/dist-packages/pycuda/cuda /tmp/tmpse323ibf/kernel.cu

# 2.修改编译器命令
vim /usr/lib/python3/dist-packages/pycuda/compiler.py

将这句：（207行） arch = "sm_%d%d" % Context.get_device().compute_capability()

改为： arch = "sm_70"

```

# 二维DCT反变换
![](https://images0.cnblogs.com/blog/661248/201408/311252416889487.png)