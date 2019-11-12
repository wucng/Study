from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    # MODEL_FILE = "lenet5.uff"
    MODEL_FILE = "tf_args.npz"
    ENGINE_FILE = "lenet5.engine"  # 保存engine文件
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"
    BATCH_SIZE = 32
    NUM_classes = 10
    DTYPE = trt.float32 # 设置数据精度

def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # tensorflow 输入格式为[N,H,W,C],而tensorrt(pytorch)要求的格式为[N,C,H,W],
    # tensorflow 权重格式[f,f,in,out] ,tensorrt(pytorch)权重格式：[out,in,f,f]
    # fc1_w = weights['fc1.weight']
    # fc1_b = weights['fc1.bias']
    # fc1_w = weights['dense/kernel:0'].tanspose([1,0]) #.transpose((3,2,0,1))
    fc1_w = np.transpose(weights['dense/kernel:0'],[1,0]).reshape(-1)
    fc1_b = weights['dense/bias:0']
    fc1 = network.add_fully_connected(input=input_tensor, num_outputs=512, \
         kernel=fc1_w, bias=fc1_b)
    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    # fc2_w = weights['fc2.weight']
    # fc2_b = weights['fc2.bias']
    # fc2_w = weights['dense_1/kernel:0'].tanspose([1,0]) #.transpose((3,2,0,1))
    fc2_w = np.transpose(weights['dense_1/kernel:0'], [1, 0]).reshape(-1)
    fc2_b = weights['dense_1/bias:0']
    fc2 = network.add_fully_connected(input=relu1.get_output(0), num_outputs=ModelData.NUM_classes, \
          kernel=fc2_w, bias=fc2_b)
    # softmax2 = network.add_activation(input=fc2.get_output(0), type=trt.ActivationType.SOFTMAX)
    softmax2=network.add_softmax(input=fc2.get_output(0))

    softmax2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax2.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_batch_size = ModelData.BATCH_SIZE  # batch size
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

def main():
    model_path=os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)
    engine_file = os.path.join(model_path, ModelData.ENGINE_FILE)

    weights=np.load(model_file)
    with build_engine(weights) as engine:
        # 将模型序列化为模型流：
        # serialized_engine = engine.serialize()
        # 序列化引擎并写入文件：
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    main()