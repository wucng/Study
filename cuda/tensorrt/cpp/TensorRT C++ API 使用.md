- [tensorrt-developer-guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

---
@[toc]
# 1.简介
TensorRT使开发人员能够导入，校准，生成和部署优化的网络。网络可以直接从Caffe导入，也可以通过UFF(`tensorflow`)或ONNX(`pytorch`)格式从其他框架导入。也可以通过实例化各个图层并直接设置参数和权重以编程方式创建它们(`tensorRT API`)。

TensorRT在所有支持的平台上提供C ++实现，并在x86上提供Python实现。

TensorRT核心库中的`关键接口`是：
- 网络定义(`Network`)
网络定义接口为应用程序提供了指定网络定义的方法。可以指定输入和输出张量，可以添加层，并且有一个用于配置每种支持的层类型的界面。以及卷积层和循环层等层类型，以及Plugin层类型都允许应用程序实现TensorRT本身不支持的功能。有关网络定义的更多信息，请参见[网络定义API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_network_definition.html)。

- 建造者(`builder`)
Builder界面允许根据网络定义创建优化的引擎。它允许应用程序指定最大批处理和工作空间大小，最小可接受的精度级别，用于自动调整的定时迭代计数以及用于量化网络以8位精度运行的接口。有关Builder的更多信息，请参见[Builder API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_builder.html)。

- 发动机(`Engine`)
Engine接口允许应用程序执行推理。它支持同步和异步执行，概要分析以及枚举和查询引擎输入和输出的绑定。单个引擎可以具有多个执行上下文，从而允许将一组训练有素的参数用于同时执行多个批次。有关引擎的更多信息，请参见[Execution API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_cuda_engine.html)。

TensorRT提供了`解析器`，用于导入经过训练的网络以创建网络定义：

- Caffe解析器
该解析器可用于解析在BVLC Caffe或NVCaffe 0.16中创建的Caffe网络。它还提供了为自定义图层注册插件工厂的功能。有关C ++ Caffe解析器的更多详细信息，请参见[NvCaffeParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvcaffeparser1_1_1_i_caffe_parser.html)或Python [Caffe解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Caffe/pyCaffe.html)。

- UFF解析器
该解析器可用于解析UFF格式的网络。它还提供了注册插件工厂并为自定义层传递字段属性的功能。有关C ++ UFF解析器的更多详细信息，请参见[NvUffParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvuffparser_1_1_i_uff_parser.html)或Python [UFF解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Uff/pyUff.html)。

- ONNX解析器
该解析器可用于解析ONNX模型。有关C ++ ONNX解析器的更多详细信息，请参见[NvONNXParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/)或Python [ONNX解析器](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)。

限制：由于ONNX格式正在快速开发中，因此您可能会遇到模型版本与解析器版本之间的版本不匹配的情况。TensorRT 5.0.0随附的ONNX解析器支持ONNX IR（中间表示）版本0.0.3，opset版本7。


# 2. C++ API 使用TensorRT
假设您是从训练有素的模型开始的。本章将介绍使用TensorRT的以下必要步骤：
- 根据模型创建TensorRT网络定义
- 调用TensorRT构建器以从网络创建优化的运行时引擎
- 序列化和反序列化引擎，以便可以在运行时快速重新创建它
- 向引擎提供数据以执行推理

# C++ API与Python API
本质上，C ++ API和Python API在满足您的需求方面应该接近相同。C++ API应该用于任何对性能有严格要求的场景，以及在安全性很重要的情况下，例如在汽车中。

Python API的主要优点是易于进行数据预处理和后处理，因为您可以使用各种库，例如`NumPy`和`SciPy`。有关Python API的更多信息，请参阅[Python API使用TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)。

## 2.1 在C ++中实例化TensorRT对象
可以通过以下两种方式之一创建引擎：
- 通过用户模型中的网络定义。在这种情况下，可以选择对引擎进行序列化并保存以供以后使用。
- 通过从磁盘读取序列化的引擎。在这种情况下，性能会更好，因为绕过了解析模型和创建中间对象的步骤。

---

使用iNetwork定义作为输入来创建可用的解析器之一：
- ONNX: `parser = nvonnxparser::createParser(*network, gLogger);`
- NVCaffe: `ICaffeParser* parser = createCaffeParser();`
- UFF: `parser = createUffParser();`

---
全局TensorRT API方法调用`createInferBuilder（gLogger）`用于创建`iBuilder`类型的对象，如图5所示。有关更多信息，请参见IBuilder类参考。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175208194.png)
图5.使用iLogger作为输入参数创建iBuilder

---
调用iBuilder的`createNetwork`的方法用于创建`iNetworkDefinition`类型的对象，如图6所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175346452.png)
图6. `createNetwork()`用于创建网络

---

`iParser`类型的对象调研为`parse()`的方法以读取模型文件并填充TensorRT网络。图7。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115174843358.png)

---
调用iBuilder的`buildCudaEngine()`的方法来创建一个`iCudaEngine`类型的对象，如图8所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175015815.png)

---
可以选择将引擎序列化并转储到文件中(生成`engine`文件)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175447492.png)

---
执行上下文用于执行推断。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175616777.png)

---
如果保留了序列化引擎并将其保存到文件中，则可以跳过上述大多数步骤。

createInferRuntime（gLogger）的全局TensorRT API方法用于创建`iRuntime`类型的对象，如图11所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115175752517.png)
有关TensorRT运行时的更多信息，请参见IRuntime类参考。 通过调用运行时方法`deserializeCudaEngine()`创建引擎(直接加载前面已经保存的`engin`文件)。

建议在创建运行时或构建器对象之前创建和配置CUDA上下文。

## 2.2 在C++中创建网络定义
使用TensorRT进行推理的第一步是根据您的模型创建一个TensorRT网络。最简单的方法是使用TensorRT解析器库导入模型，该库支持以下格式的序列化模型：

- [sampleMNIST](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#mnist_sample)（BVLC和NVCaffe）
- [sampleOnnxMNIST](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#onnx_mnist_sample)
- [sampleUffMNIST](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#mnist_uff_sample)（用于TensorFlow）

另一种选择是直接使用[TensorRT API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html)定义模型。这要求您进行少量的API调用，以定义网络图中的每一层，并为模型的训练参数实现自己的导入机制。

输入和输出张量也必须给定名称（使用`ITensor::setName()`）

TensorRT网络定义的一个重要方面是它包含指向模型权重的指针，模型权重由构建器复制到优化引擎中。如果网络是通过解析器创建的，则解析器将拥有权重占用的内存，因此，在运行构建器之后，才应删除解析器对象。

### 2.2.1 使用`C++ API`从头开始创建网络定义
除了使用解析器之外，您还可以通过网络定义API 将网络直接定义到TensorRT。这种情况假设在网络创建过程中主机内存中的每层权重已准备好传递给TensorRT。

在下面的示例中，我们将创建一个具有输入，卷积，池化，完全连接，激活和SoftMax层的简单网络。要查看全部的代码，请参阅[sampleMNISTAPI](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#mnistapi_sample)设在`/usr/src/tensorrt/samples/sampleMNISTAPI` 目录。

- 1.创建构建器和网络(`builder and Network`)：
```c
IBuilder* builder = createInferBuilder(gLogger);
INetworkDefinition* network = builder->createNetwork();
```
- 2.将输入层和输入尺寸添加到网络。一个网络可以有多个输入，尽管在此示例中只有一个：
```c
auto data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
```
- 3.添加具有隐藏层输入节点，步幅和权重的卷积层以进行滤波和偏置。为了从图层中检索张量参考，我们可以使用：
```c
layerName->getOutput(0)
auto conv1 = network->addConvolution(*data->getOutput(0), 20, DimsHW{5, 5}, weightMap["conv1filter"], weightMap["conv1bias"]);
conv1->setStride(DimsHW{1, 1});
```

注意：传递到TensorRT层的权重在主机内存中。

- 4.添加池层：
```c
auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
pool1->setStride(DimsHW{2, 2});
```
- 5.添加FullyConnected和Activation层：
```c
auto ip1 = network->addFullyConnected(*pool1->getOutput(0), 500, weightMap["ip1filter"], weightMap["ip1bias"]);
auto relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
```
- 6.添加SoftMax层以计算最终概率并将其设置为输出：
```c
auto prob = network->addSoftMax(*relu1->getOutput(0));
prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
```

- 7.标记输出：
```c
network->markOutput(*prob->getOutput(0));
```

### 2.2.2 在C ++中使用解析器导入模型
要使用`C++ Parser API`导入模型，您将需要执行以下高级步骤：
- 1.创建TensorRT构建器和网络。
```c
IBuilder* builder = createInferBuilder(gLogger);
nvinfer1::INetworkDefinition* network = builder->createNetwork();
```
有关如何创建记录器的示例，请参见在C ++中实例化TensorRT对象。
- 2.为特定格式创建TensorRT解析器。
```c
// ONNX
auto parser = nvonnxparser::createParser(*network,
        gLogger);

// UFF
auto parser = createUffParser();

// NVCaffe
ICaffeParser* parser = createCaffeParser();
```
- 3.使用解析器来解析导入的模型并填充网络
```c
parser->parse(args);
```
### 2.2.3 使用`C++ Parser API` 导入Caffe模型
以下步骤说明了如何使用`C++ Parser API `导入Caffe模型。有关更多信息，请参见sampleMNIST
```c
// 1.创建构建器和网络：
IBuilder* builder = createInferBuilder(gLogger);
INetworkDefinition* network = builder->createNetwork();

// 2.创建Caffe解析器：
ICaffeParser* parser = createCaffeParser();

// 3.解析导入的模型：
const IBlobNameToTensor* blobNameToTensor = parser->parse("deploy_file" , "modelFile", *network, DataType::kFLOAT);
/*
这将从Caffe模型填充TensorRT网络。最后一个参数指示解析器生成权重为32位浮点数的网络。使用`DataType::kHALF` 会生成一个具有16位权重的模型。

除了填充网络定义之外，解析器还返回一个字典，该字典将Caffe Blob名称映射到TensorRT张量。与Caffe不同，TensorRT网络定义没有就地操作的概念。当Caffe模型使用就地操作时，字典中返回的 TensorRT张量对应于对该Blob的最后一次写入。例如，如果卷积写入blob，然后是就地ReLU，则该blob的名称将映射到TensorRT 张量是ReLU的输出。
*/

// 4.指定网络的输出
for (auto& s : outputs)
    network->markOutput(*blobNameToTensor->find(s.c_str()));
```
### 2.2.4. 使用`C++ UFF Parser API`导入TensorFlow Model
注意：对于新项目，建议使用TensorFlow-TensorRT集成作为将TensorFlow网络转换为使用TensorRT进行推理的方法。有关集成说明，请参阅[将TensorFlow与TensorRT集成](https://docs.nvidia.com/deeplearning/frameworks/)及其发行说明。

从TensorFlow框架导入需要将TensorFlow模型转换为中间格式`UFF`（通用框架格式）。有关转换的更多信息，请参见[将冻结图转换为UFF](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#samplecode3)。
以下步骤说明了如何使用C++ Parser API 导入TensorFlow模型。有关UFF导入的更多信息，请参见[sampleUffMNIST](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#mnist_uff_sample)。
```c
// 1.创建构建器和网络：
IBuilder* builder = createInferBuilder(gLogger);
INetworkDefinition* network = builder->createNetwork();

// 2.创建UFF解析器：
IUFFParser* parser = createUffParser();

// 3.向UFF解析器声明网络输入和输出：
parser->registerInput("Input_0", DimsCHW(1, 28, 28), UffInputOrder::kNCHW);
parser->registerOutput("Binary_3");
// 注意：TensorRT期望输入张量按CHW顺序排列。从TensorFlow导入时，请确保输入张量处于要求的顺序，如果不是，则将其转换为CHW。

// 4.解析导入的模型以填充网络：
parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT);
```

### 2.2.5.使用`C++ Parser API` 导入ONNX模型
限制：由于ONNX格式正在快速开发中，因此您可能会遇到模型版本与解析器版本之间的版本不匹配的情况。TensorRT 5.0.0随附的ONNX解析器支持ONNX IR（中间表示）版本0.0.3，opset版本7。

以下步骤说明了如何使用C ++ Parser API 导入ONNX模型。有关ONNX导入的更多信息，请参见`sampleOnnxMNIST`。

```c
// 1.创建ONNX解析器。解析器使用辅助配置管理SampleConfig 对象，将输入参数从示例可执行文件传递到解析器对象
nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();
//Create Parser
nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);

// 2.摄取模型：
parser->parse(onnx_filename, DataType::kFLOAT);

// 3.将模型转换为TensorRT网络：
parser->convertToTRTNetwork();

// 4.从模型获取网络：
nvinfer1::INetworkDefinition* trtNetwork = parser->getTRTNetwork();
```

## 2.3 用C++构建引擎(Engine)
两个特别重要的属性是`最大批处理大小`和`最大工作空间大小`。

- 最大批次大小指定TensorRT将优化的批次大小。在运行时，可以选择较小的批次大小(`batch_size`)。
- 层算法通常需要临时工作空间。此参数限制网络中任何层可以使用的最大大小。如果提供的预留空间不足，则TensorRT可能无法找到给定层的实现。

```c
// 1.使用构建器对象构建引擎：
builder->setMaxBatchSize(maxBatchSize);
builder->setMaxWorkspaceSize(1 << 20);
ICudaEngine* engine = builder->buildCudaEngine(*network);

// 构建引擎后，TensorRT会复制权重。
// 2.如果使用网络，构建器和解析器，需释放它。
engine->destroy();
network->destroy();
builder->destroy();
```

## 2.4在C++中序列化模型(保存engine文件)
构建可能需要一些时间，因此一旦构建引擎，您通常将需要对其进行序列化以供以后使用。

==注意:== 序列化引擎不可跨平台或TensorRT 版本移植。引擎特定于它们所构建的确切GPU模型（除了平台和TensorRT版本）。

序列化
```c
#if(1)
	// 保存engine文件（下次直接加载engine文件即可）
	IHostMemory* serializedModel = mEngine->serialize();

	// 保存路径
	std::string cache_path = "../cached_model.engine";
	std::ofstream serialize_output_stream;

	// 将序列化的模型结果拷贝至serialize_str字符串
	std::string serialize_str;
	serialize_str.resize(serializedModel->size());
	memcpy((void*)serialize_str.data(), serializedModel->data(), serializedModel->size());

	// 将serialize_str字符串的内容输出至cached_model.bin文件
	serialize_output_stream.open(cache_path);
	serialize_output_stream << serialize_str;
	serialize_output_stream.close();

	serializedModel->destroy(); // 释放内存
#endif
```

创建运行时对象以反序列化：
```c
#if(0) // 从上面保存的engine文件中加载，反序列化得到mEngine
	iRuntime* _runtime = createInferRuntime(gLogger);

	// 从cached_model.engine文件中读取序列化的结果
	std::string cache_path = "../cached_model.engine";
	std::ifstream fin(cached_path);

	// 将文件中的内容读取至cached_engine字符串

	std::string cached_engine = "";
	while (fin.peek() != EOF) { // 使用fin.peek()防止文件读取时无限循环
		std::stringstream buffer;
		buffer << fin.rdbuf();
		cached_engine.append(buffer.str());
	}
	fin.close();

	// 将序列化得到的结果进行反序列化，以执行后续的inference
	mEngine = _runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), &_plugin_factory);

#endif
```
最后一个参数是使用自定义层的应用程序的插件层工厂。有关更多信息，请参阅使用[自定义层扩展TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending)。

## 2.5 在C++中执行推理
有了引擎，以下步骤说明了如何在C ++中执行推理。
```c
// 1.创建一些空间来存储中间激活值。由于引擎保留了网络定义和训练有素的参数，因此需要额外的空间。这些是在执行上下文中保存的：
IExecutionContext *context = engine->createExecutionContext();

/*
引擎可以具有多个执行上下文，从而允许将一组权重用于多个重叠的推理任务。例如，您可以使用一个引擎和每个流一个上下文来处理并行CUDA流中的图像。每个上下文将在与引擎相同的GPU上创建。
*/
// 2.使用输入和输出Blob名称来获取相应的输入和输出索引：
int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

// 3.使用这些索引，设置一个缓冲区数组，该数组指向GPU上的输入和输出缓冲区：
void* buffers[2];
buffers[inputIndex] = inputbuffer;
buffers[outputIndex] = outputBuffer;

// 4.TensorRT执行通常是异步的，因此 入队CUDA流上的内核：
context.enqueue(batchSize, buffers, stream, nullptr);
```

## 2.6 C++中的内存管理
ensorRT提供两种机制，以允许应用程序对设备内存进行更多控制。
默认情况下，创建 `IExecutionContext`，分配了持久性设备内存来保存激活数据。为避免这种分配，请致电 `createExecutionContextWithoutDeviceMemory`。然后由应用程序负责调用 `IExecutionContext :: setDeviceMemory（）`提供运行网络所需的内存。内存块的大小由以下方式返回 `ICudaEngine :: getDeviceMemorySize（）`。

此外，应用程序可以通过实现以下功能来提供自定义分配器，以便在构建和运行时使用： IGpu分配器接口。接口实现后，调用
```c
setGpuAllocator(&allocator);
```
在 iBuilder的 要么 运行时接口。然后将通过此接口分配和释放所有设备内存。


