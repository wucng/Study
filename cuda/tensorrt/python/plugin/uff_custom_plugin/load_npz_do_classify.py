"""
tensorrt api加载plugin
保存成engin文件，再加载会报错
"""
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import ctypes
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'build/libclipplugin.so'
)
ctypes.CDLL(CLIP_PLUGIN_LIBRARY)
# lib = ctypes.cdll.LoadLibrary(CLIP_PLUGIN_LIBRARY)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            clipMin_field=trt.PluginField("clipMin", np.array([0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            clipMax_field=trt.PluginField("clipMax", np.array([6.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)

            field_collection = trt.PluginFieldCollection([clipMin_field, clipMax_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin

class ModelData(object):
    # MODEL_FILE = "lenet5.uff"
    MODEL_FILE = "tf_args.npz"
    ENGINE_FILE = "lenet5.engine"  # 保存engine文件
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (1, 28, 28)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    BATCH_SIZE = 32
    NUM_classes = 10
    DTYPE = trt.float16 # 设置数据精度

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
    # relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    # plguin relu6
    relu1 = network.add_plugin_v2(inputs=[fc1.get_output(0)], plugin=get_trt_plugin("CustomClipPlugin"))

    # fc2_w = weights['fc2.weight']
    # fc2_b = weights['fc2.bias']
    # fc2_w = weights['dense_1/kernel:0'].tanspose([1,0]) #.transpose((3,2,0,1))
    fc2_w = np.transpose(weights['OutputLayer/kernel:0'], [1, 0]).reshape(-1)
    fc2_b = weights['OutputLayer/bias:0']
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

import tensorflow as tf
def load_normalized_test_case(pagelocked_buffer):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    np.copyto(pagelocked_buffer, x_test[:ModelData.BATCH_SIZE].ravel())

    # [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    # img = np.array(Image.open(test_case_path)).ravel()
    # np.copyto(pagelocked_buffer, 1.0 - img / 255.0)

    return y_test[:ModelData.BATCH_SIZE]

def main():
    model_path=os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)
    engine_file = os.path.join(model_path, ModelData.ENGINE_FILE)

    weights=np.load(model_file)
    with build_engine(weights) as engine:
        """
        # 将模型序列化为模型流：
        # serialized_engine = engine.serialize()
        # 序列化引擎并写入文件：
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())

    # 从文件中读取引擎并反序列化：
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        """
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            # case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
            case_num = load_normalized_test_case(pagelocked_buffer=inputs[0].host)

            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs,
                                           outputs=outputs, stream=stream, batch_size=ModelData.BATCH_SIZE)
            output = np.reshape(output, [-1, ModelData.NUM_classes])  # 默认是一行的，展成原来的形状
            pred = np.argmax(output, -1)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))

    print("\ninference success!\n")


if __name__ == '__main__':
    main()
