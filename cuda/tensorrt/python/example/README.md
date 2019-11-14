- [数据tomato](https://www.kaggle.com/noulam/tomato)
- [resnet18 50网络结构以及pytorch实现代码](https://www.jianshu.com/p/085f4c8256f1)

描述：分为训练集与验证集，共10个类别，每张图片大小都为256 x 256
---
# 步骤
## 1.使用`pytorch`训练一个分类器

## 2.使用`tensorrt API`做推理
- 1、使用pytorch内置的`resnet18`
	- [pytorch 推理](./inference.py)结果：`Accuracy: 4520/4585 (99%)`
	- [tensorRT API推理](./loadEngineDoClassify.py)结果：`test acc:(4207 / 4585) 0.918`
	- [onnx 推理](./onnx/loadEngineDoClassify.py)结果：`test acc:(4520 / 4585) = 0.986`


- 2、自定义重写的`resnet18`
	- [pytorch 推理](./pythonAPI/inference.py)结果：`Accuracy: 1745/4585 (38%)`
	- [tensorRT API推理](./pythonAPI/loadEngineDoClassify.py)结果：`test acc:(1742 / 4585) = 0.380`


==注:== 
直接使用`tensorRT API` 解析pytorch内置的`resnet18`(内置模型很多细节没法实现)，导致结果偏差很大，可以使用onnx来解析

直接使用`tensorRT API` 解析自定义模型(模型细节不存在高度隐藏)，偏差不大

使用onnx解析时，如果版本不匹配会导致解析失败：
```python
cuda 10.0 (cudnn 7.6.2)
TensorRT 6.0.1.5
pytorch 1.3.1(torchvision 0.4.2) # 与TensorRT版本不匹配，`tensorrt`的`OnnxParser`解析报错
pytorch 1.2.0(torchvision 0.4.0) # `tensorrt`的`OnnxParser`解析成功
```

## 3.使用`bottle`发布为一个服务器

![](https://upload-images.jianshu.io/upload_images/15074510-c6806cfdf2a88fc4.png?imageMogr2/auto-orient/strip|imageView2/2/w/999/format/webp)

![](https://upload-images.jianshu.io/upload_images/15074510-ef99906b7dbbd4fd.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
