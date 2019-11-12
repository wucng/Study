pytorch模型保存成onnx格式,再使用tensorrt解析转成engin

1、直接使用`torch.onnx.export`转成onnx文件
`tensorrt`的`OnnxParser`解析==会出错== (版本问题，tensorrt与pytorch版本不匹配)

```python
cuda 10.0 (cudnn 7.6.2)
TensorRT 6.0.1.5
pytorch 1.3.1(torchvision 0.4.2) # 与TensorRT版本不匹配，`tensorrt`的`OnnxParser`解析报错
pytorch 1.2.0(torchvision 0.4.0) # `tensorrt`的`OnnxParser`解析成功
```

2、解决方法
- I.使用[onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)解析
- II.使用`onnx`将torch模型转成onnx格式，再使用`tensorrt`的`OnnxParser`解析 (完全可以直接解析npz文件，再使用tensorrt api转成engin)
