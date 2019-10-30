- https://docs.nvidia.com/cuda/cublas/index.html
- https://docs.nvidia.com/cuda/index.html
- https://developer.nvidia.com/gpu-accelerated-libraries
- https://developer.nvidia.com/cublas

---
# cuBLAS(矩阵运算库)
![](https://img-blog.csdnimg.cn/20191025220607189.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70&ynotemdtimestamp=1572337582480)

 cuBLAS中能用于运算矩阵乘法的函数有4个，分别是 `cublasSgemm`（单精度实数）、`cublasDgemm`（双精度实数）、`cublasCgemm`（单精度复数）、`cublasZgemm`（双精度复数），它们的定义（在 cublas_v2.h 和 cublas_api.h 中）如下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191029171513938.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191029171529118.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)