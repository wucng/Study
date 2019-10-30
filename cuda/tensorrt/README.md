- https://github.com/LitLeo/TensorRT_Tutorial
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html

---
# 一、TensorRT理论解释
TensorRT所做的优化，总结下来主要有这么几点：
- 第一，也是最重要的，它把==一些网络层进行了合并==。大家如果了解GPU的话会知道，在GPU上跑的函数叫Kernel，TensorRT是存在Kernel的调用的。在绝大部分框架中，比如一个卷积层、一个偏置层和一个reload层，这三层是需要调用三次cuDNN对应的API，但实际上这三层的实现完全是可以合并到一起的，TensorRT会对一些可以合并网络进行合并；再比如说，目前的网络一方面越来越深，另一方面越来越宽，可能并行做若干个相同大小的卷积，这些卷积计算其实也是可以合并到一起来做的。

- 第二，比如在concat这一层，比如说这边计算出来一个1×3×24×24，另一边计算出来1×5×24×24，concat到一起，变成一个1×8×24×24的矩阵，这个叫concat这层这其实是完全没有必要的，因为TensorRT完全可以实现直接接到需要的地方，不用专门做concat的操作，所以这一层也可以取消掉。
- 第三，Kernel可以根据不同的batch size 大小和问题的复杂程度，去选择最合适的算法，TensorRT预先写了很多GPU实现，有一个自动选择的过程。

- 第四，不同的batch size会做tuning。

- 第五，不同的硬件如P4卡还是V100卡甚至是嵌入式设备的卡，TensorRT都会做优化，得到优化后的engine。

下图是一个原始的GoogleNet的一部分，首先input后会有多个卷积，卷积完后有Bias和ReLU，结束后将结果concat（连接拼接）到一起，得到下一个input。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy90TUp0ZmdJSWliV0wwNWljQXIwSEROMHFVTzR6TTFjOEgwNExURWxuRGM4R0NBc3cwalU4OWp0Tzl0dmt5MHo1eTV4VzZkYVByTnl5YlVoaWFha1Nsc1lMQS82NDA?x-oss-process=image/format,png)
以上的整个过程可以做些什么优化呢？首先是convolution, Bias和ReLU这三个操作可以合并成CBR，合并后的结果如下所示，其中包含四个1×1的CBR，一个3×3的CBR和一个5×5的CBR。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy90TUp0ZmdJSWliV0wwNWljQXIwSEROMHFVTzR6TTFjOEgwWGlhM0prbUhLTGpTaWExaWF4ajh2bXd2N29lNEd2c0ZkeXJIVjRsRFVjVDZ6MlRRRXowM013MjBBLzY0MA?x-oss-process=image/format,png)
接下来可以继续合并三个相连的1×1的CBR为一个大的1×1的CBR（如下图），这个合并就可以更好地利用GPU。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy90TUp0ZmdJSWliV0wwNWljQXIwSEROMHFVTzR6TTFjOEgwcGJkenR1VW0zT0NaR0g4MXlyeDNPWmRSUndpYlpibElKS0R3VEJ3TWF1b3BpYnN0MmxOQldCUVEvNjQw?x-oss-process=image/format,png)
继而concat层可以消除掉，直接连接到下一层的next input（如下图）。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy90TUp0ZmdJSWliV0wwNWljQXIwSEROMHFVTzR6TTFjOEgwVENBcVozU2s4MjBEN3NHRmlhbjEzWlZ2U2R5aDhnN0piQXlDaHRuNHJzSzltTEFuSkZpY0ZiTFEvNjQw?x-oss-process=image/format,png)
另外还可以做并发（Concurrency），如下图左半部分（max pool和1×1 CBR）与右半部分（大的1×1 CBR，3×3 CBR和5×5 CBR）彼此之间是相互独立的两条路径，本质上是不相关的，可以在GPU上通过并发来做，来达到的优化的目标。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy90TUp0ZmdJSWliV0wwNWljQXIwSEROMHFVTzR6TTFjOEgwTENtS1RFZ2QzUDdYUk90NFh0NG1zOVM2aWJKaWNlUlZqcFMyNFlQNzdXQkpOQTc0Z2hBaFVRRHcvNjQw?x-oss-process=image/format,png)
# 二、TensorRT高级特征介绍
## 1. 插件支持
首先TensorRT是支持插件（Plugin）的，或者前面提到的Customer layer的形式，也就是说我们在某些层TensorRT不支持的情况下，最主要是做一些检测的操作的时候，很多层是该网络专门定义的，TensorRT没有支持，需要通过Plugin的形式自己去实现。实现过程包括如下两个步骤：
- 首先需要重载一个IPlugin的基类，生成自己的Plugin的实现，告诉GPU或TensorRT需要做什么操作，要构建的Plugin是什么样子，其实就是类似于开发一个应用软件的插件，需要在上面实现什么功能。

- 其次要将插件添加到合适的位置，在这里是要添加到网络里去。

## 2. 低精度支持
低精度指的是之前所说过的FP16和INT8，其中FP16主要是Pascal P100和V100（tensor core）这两张卡支持；而INT8主要针对的是 P4和P40这两张卡，P4是专门针对线上做推断（Inference）的小卡，和IPhone手机差不多大，75瓦的一张卡，功耗和性能非常好。

## 3. Python接口和更多的框架支持
TensorRT目前支持Python和C++的API，刚才也介绍了如何添加，Model importer（即Parser）主要支持Caffe和Uff，其他的框架可以通过API来添加，如果在Python中调用pyTouch的API，再通过TensorRT的API写入TensorRT中，这就完成了一个网络的定义。



TensorRT去做推断（Inference）的时候是不再需要框架的，用Caffe做推断（Inference）需要Caffe这个框架，TensorRT把模型导进去后是不需要这个框架的，Caffe和TensorFlow可以通过Parser来导入，一开始就不需要安装这个框架，给一个Caffe或TensorFlow模型，完全可以在TensorRT高效的跑起来。

# 三、用户自定义层
使用插件创建用户自定义层主要分为两个步骤：
- 第一步是创建使用IPlugin接口创建用户自定义层，IPlugin是TensorRT中预定义的C++抽象类，用户需要定义具体实现了什么。
- 第二步是将创建的用户自定义层添加到网络中，如果是Caffe的模型，不支持这一层，将名字改成IPlugin是可以识别的，当然还需要一些额外的操作，说明这一层的操作是对应哪个Plugin的实现；而对于Uff是不支持Plugin的Parser，也就是说TensorFlow的模型中有一个Plugin的话，是不能从模型中识别出来的，这时候需要用到addPlugin()的方法去定义网络中Plugin的相关信息。

IPlugin接口中需要被重载的函数有以下几类：

