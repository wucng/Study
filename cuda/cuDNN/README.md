- https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html
- https://developer.nvidia.com/cuDNN
---
# 1.简介
NVIDIA®cuDNN是用于深度神经网络的GPU加速的原语库。 它提供了DNN应用程序中经常出现的例程的高度优化的实现：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030135113109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)

# 2.概述
使用`cuDNN`的应用程序必须通过调用`cudnnCreate()`来初始化库上下文的句柄。 该句柄被显式传递给对GPU数据进行操作的每个后续库函数。 一旦应用程序使用cuDNN完成，它就可以使用`cudnnDestroy()`释放与库句柄关联的资源。 当使用多个主机线程，GPU和CUDA流时，该方法允许用户显式控制库的功能。

例如，应用程序可以使用`cudaSetDevice()`将不同的设备与不同的主机线程相关联，并且在每个主机线程中，使用唯一的cuDNN句柄将库调用定向到与其关联的设备。 因此，使用不同句柄进行的cuDNN库调用将自动在不同设备上运行。

假定与特定cuDNN上下文关联的设备在相应的`cudnnCreate()`和`cudnnDestroy()`调用之间保持不变。 为了使cuDNN库在同一主机线程中使用其他设备，应用程序必须通过调用`cudaSetDevice()`来设置要使用的新设备，然后通过调用`cudnnCreate()`创建与新设备关联的另一个cuDNN上下文。

## 2.2.卷积公式
本节描述了在cuDNN卷积函数中实现的各种卷积公式。

下表中描述的卷积项适用于随后的所有卷积公式。

卷积参数表

## 2.3.符号
使用4-D张量描述符来定义具有4个字母的2D图像批次的格式：N，C，H，W，分别代表批次大小，特征图数量，高度和宽度。 字母按步幅降序排列。 常用的4-D张量格式为：
```c
NCHW
NHWC
CHWN
```
5-D张量描述符用于定义5个字母的3D图像批处理的格式：N，C，D，H，W分别表示批处理大小，特征图的数量，深度，高度和宽度 。 字母以降序排列。 常用的5-D张量格式称为：
```c
NCDHW
NDHWC
CDHWN
```
