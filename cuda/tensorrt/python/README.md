- https://github.com/NVIDIA/TensorRT/tree/master/samples
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html
- `TensorRT-6.0.1.5/samples/python` ([TensorRT-6.0.1.5](https://developer.nvidia.com/tensorrt)安装包)

---

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191112133154539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
# 1、推荐使用`tensorrt api`
- 优点：过程可控，设置数据输入精度(默认是`float32`)，解析速度快
- 缺点：搭建麻烦，费时

注：使用`tensorRT API`时：
- `tensorRT` 与`pytorch`的格式一样，输入格式：`[N,C,H,W]`，权重格式：`[out,in,fh,fw]`
- `tensorflow`默认格式为，输入格式：`[N,H,W,C]`，权重格式：`[fh,fw,in,out]`
- 使用`tensorrt Api`加载`tensorflow`权重时，需先转置转成`tensorrt`格式，最后必须加上`reshape(-1)` （而`pytorch`的权重解析时可以不加`reshape(-1)`）

# 2、使用`trt.UffParser` 或 `trt.OnnxParser` 解析
- 优点：搭建方便
- 缺点：过程不可控，无法设置输入精度，解析数度慢

# 3、使用自定义方式（不推荐）
- I.使用`numpy`解析
- II.使用`cupy`(numpy的cuda版)解析
- III.使用`pycuda`解析 
