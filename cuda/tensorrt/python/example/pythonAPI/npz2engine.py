import model
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32 #trt.float16
    NP_DTYPE = np.float32
    BATCH_SIZE = 32

def trt_bn(network,input_size,name,weights,dtype,belta=1e-3):
    g0 = weights[name+'.weight']  # .reshape(-1)
    b0 = weights[name+'.bias']  # .reshape(-1)
    m0 = weights[name+'.running_mean']  # .reshape(-1)
    v0 = weights[name+'.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + belta)
    shift0 = -m0 / np.sqrt(v0 + belta) * g0 + b0
    power0 = np.ones(len(g0), dtype=dtype)
    bn1 = network.add_scale(input=input_size, mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,
                            power=power0)
    return bn1

def trt_conv(network,input_size,name,weights,dtype,kernel_shape,stride,padding):
    conv1_w = weights[name+'.weight']
    conv1_b = weights[name+'.bias'] if name+'.bias' in weights else np.zeros([conv1_w.shape[0]], dtype=dtype)
    conv1 = network.add_convolution(input=input_size, num_output_maps=conv1_w.shape[0],
                                    kernel_shape=kernel_shape, kernel=conv1_w, bias=conv1_b)
    conv1.stride = stride
    conv1.padding = padding

    return conv1

def trt_pool(network,input_size,type,pool_size,stride,padding=(0,0)):
    pool1 = network.add_pooling(input=input_size, type=type, window_size=pool_size)
    pool1.stride = stride
    pool1.padding = padding
    return pool1

def layer_common(network,weights,input_size,dtype,
                 conv_config=[],
                 bn_name="",
                 activation_type=None,
                 pool_type=trt.PoolingType.MAX,
                 pool_config=[]):
    """
    conv + batch_norm + [relu] +maxpool
    :return:
    """
    conv1 = trt_conv(network, input_size, conv_config[0], weights, dtype,
                              conv_config[1], conv_config[2],conv_config[3])
    bn1 = trt_bn(network, conv1.get_output(0), bn_name, weights, dtype)
    if activation_type!=None:
        relu1 = network.add_activation(input=bn1.get_output(0), type=activation_type)
        maxpool1 = trt_pool(network, relu1.get_output(0), pool_type, pool_config[0],
                                     pool_config[1], pool_config[2])
    else:
        maxpool1 = trt_pool(network, bn1.get_output(0), pool_type, pool_config[0],
                            pool_config[1], pool_config[2])

    return maxpool1

def bottleNet(network,weights,input_size,dtype,
              conv1_config=[],
              bn1_name="",relu1_type=trt.ActivationType.RELU,
              conv2_config=[],
              bn2_name="",relu2_type=trt.ActivationType.RELU,
              downsample=[]):
    """
    resnet 瓶颈层
    """
    conv1 = trt_conv(network, input_size, conv1_config[0],
                                       weights, dtype, conv1_config[1], conv1_config[2], conv1_config[3])
    bn1 = trt_bn(network, conv1.get_output(0), bn1_name, weights,dtype)

    relu1 = network.add_activation(input=bn1.get_output(0),type=relu1_type)
    # ----------------------------------------------------------------------------------------------
    conv2 = trt_conv(network, relu1.get_output(0), conv2_config[0],
                                       weights, dtype, conv2_config[1], conv2_config[2], conv2_config[3])
    bn2 = trt_bn(network, conv2.get_output(0),bn2_name, weights,dtype)

    # 是否需要做downsample
    if len(downsample)>0:
        downsample_conv = trt_conv(network, input_size,downsample[0],
                           weights, dtype, downsample[1], downsample[2], downsample[3])
        downsample_bn = trt_bn(network, downsample_conv.get_output(0),
                             downsample[4], weights, dtype)

        backbone_layer1 = network.add_elementwise(input1=downsample_bn.get_output(0),
                                                  input2=bn2.get_output(0),
                                                  op=trt.ElementWiseOperation.SUM)
    else:
        backbone_layer1 = network.add_elementwise(input1=input_size,
                                                  input2=bn2.get_output(0),
                                                  op=trt.ElementWiseOperation.SUM)

    relu2 = network.add_activation(input=backbone_layer1.get_output(0),type=relu2_type)

    return relu2

# resnet 18
def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # ----------------layer0-------------------------------------------------
    maxpool1=layer_common(network,weights,input_tensor,ModelData.NP_DTYPE,
                                   ["layer0.conv1",(7, 7), (2, 2),(3, 3)],
                                   'layer0.bn1',trt.ActivationType.RELU,
                                   trt.PoolingType.MAX,[(3, 3), (2, 2), (1, 1)])

    # --------------------layer1.0---------------------------------------------
    layer1_0=bottleNet(network,weights,maxpool1.get_output(0),ModelData.NP_DTYPE,
                                ['layer1_0.conv1',(3, 3), (1, 1), (1, 1)],
                                'layer1_0.bn1',trt.ActivationType.RELU,
                                ['layer1_0.conv2',(3, 3), (1, 1), (1, 1)],
                                'layer1_0.bn2',
                                trt.ActivationType.RELU)

    # ----------------layer1.1-------------------------------------------------
    layer1_1=bottleNet(network,weights,layer1_0.get_output(0),ModelData.NP_DTYPE,
                                ['layer1_1.conv1',(3, 3), (1, 1), (1, 1)],
                                'layer1_1.bn1',trt.ActivationType.RELU,
                                ['layer1_1.conv2',(3, 3), (1, 1), (1, 1)],
                                'layer1_1.bn2',trt.ActivationType.RELU,
                                )

    # ----------------layer2.0-------------------------------------------------
    layer2_0 = bottleNet(network, weights, layer1_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer2_0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer2_0.bn1', trt.ActivationType.RELU,
                                  ['layer2_0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer2_0.bn2', trt.ActivationType.RELU,
                                  ["layer2_0.downsample.0",(1, 1), (2, 2), (0, 0),
                                   "layer2_0.downsample.1"]
                                  )

    # ----------------layer2.1-------------------------------------------------
    layer2_1 = bottleNet(network, weights, layer2_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer2_1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer2_1.bn1', trt.ActivationType.RELU,
                                  ['layer2_1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer2_1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------layer3.0-------------------------------------------------
    layer3_0 = bottleNet(network, weights, layer2_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer3_0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer3_0.bn1', trt.ActivationType.RELU,
                                  ['layer3_0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer3_0.bn2', trt.ActivationType.RELU,
                                  ["layer3_0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "layer3_0.downsample.1"]
                                  )

    # ----------------layer3.1-------------------------------------------------
    layer3_1 = bottleNet(network, weights, layer3_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer3_1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer3_1.bn1', trt.ActivationType.RELU,
                                  ['layer3_1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer3_1.bn2', trt.ActivationType.RELU,
                                  )

    # ----------------layer4.0-------------------------------------------------
    layer4_0 = bottleNet(network, weights, layer3_1.get_output(0), ModelData.NP_DTYPE,
                                  ['layer4_0.conv1', (3, 3), (2, 2), (1, 1)],
                                  'layer4_0.bn1', trt.ActivationType.RELU,
                                  ['layer4_0.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer4_0.bn2', trt.ActivationType.RELU,
                                  ["layer4_0.downsample.0", (1, 1), (2, 2), (0, 0),
                                   "layer4_0.downsample.1"]
                                  )

    # ----------------layer4.1-------------------------------------------------
    layer4_1 = bottleNet(network, weights, layer4_0.get_output(0), ModelData.NP_DTYPE,
                                  ['layer4_1.conv1', (3, 3), (1, 1), (1, 1)],
                                  'layer4_1.bn1', trt.ActivationType.RELU,
                                  ['layer4_1.conv2', (3, 3), (1, 1), (1, 1)],
                                  'layer4_1.bn2', trt.ActivationType.RELU,
                                  )

    # --------layer5---------------------------------
    layer5=layer_common(network, weights, layer4_1.get_output(0), ModelData.NP_DTYPE,
                 ["layer5.conv1", (1, 1), (1, 1), (0, 0)],
                 "layer5.bn1", trt.ActivationType.RELU, trt.PoolingType.AVERAGE,
                 [(1, 1), (7, 7), (0, 0)])

    # add softmax or not
    softmax_layer=network.add_softmax(layer5.get_output(0))

    softmax_layer.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax_layer.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_batch_size = ModelData.BATCH_SIZE
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def main():
    _, _ = common.find_sample_data(description="Runs an MNIST network using a PyTorch model", subfolder="mnist")
    # Train the PyTorch model
    # mnist_model = model.MnistModel()
    # mnist_model.learn()
    # weights = mnist_model.get_weights()
    weights=np.load("model.npz")
    # Do inference with TensorRT.
    engine_file="model.engine"
    with build_engine(weights) as engine:
        # 将模型序列化为模型流：
        # serialized_engine = engine.serialize()
        # 序列化引擎并写入文件：
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    main()