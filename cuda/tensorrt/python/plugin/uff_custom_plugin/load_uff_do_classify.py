"""保存成engin文件，再加载会报错"""

import sys
import os
import ctypes
from random import randint

from PIL import Image
import numpy as np
import tensorflow as tf

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import graphsurgeon as gs
import uff

# ../common.py
# sys.path.insert(1,
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         os.pardir
#     )
# )
# import common

# import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
import common

# lenet5.py
import lenet5


# MNIST dataset metadata
# MNIST_IMAGE_SIZE = 28
# MNIST_CHANNELS = 1
# MNIST_CLASSES = 10

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path where clip plugin library will be built (check README.md)
CLIP_PLUGIN_LIBRARY = os.path.join(
    WORKING_DIR,
    'build/libclipplugin.so'
)

# Path to which trained model will be saved (check README.md)
MODEL_PATH = os.path.join(
    WORKING_DIR,
    'models/trained_lenet5.pb'
)

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (1, 28, 28)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    # OUTPUT_SHAPE = (28, )
    DATA_TYPE = trt.float32
    ENGINE_FILE = "lenet5.engine"  # 保存engine文件
    BATCH_SIZE=32
    NUM_classes=10


# Generates mappings from unsupported TensorFlow operations to TensorRT plugins
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_relu6 = gs.create_plugin_node(name="trt_relu6", op="CustomClipPlugin", clipMin=0.0, clipMax=6.0)
    namespace_plugin_map = {
        ModelData.RELU6_NAME: trt_relu6
    }
    return namespace_plugin_map

# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path

# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    if not os.path.exists(output_uff_path):
        uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            [ModelData.OUTPUT_NAME],
            output_filename=output_uff_path,
            text=True
        )
    return output_uff_path

# Builds TensorRT Engine
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_batch_size = ModelData.BATCH_SIZE  # batch size
        builder.max_workspace_size = common.GiB(1)
        uff_path = model_to_uff(model_path)
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(uff_path, network)

        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer. Returns loaded test case label.
def load_normalized_test_case2(pagelocked_buffer):
    _, _, x_test, y_test = lenet5.load_data()
    num_test = len(x_test)
    case_num = randint(0, num_test-1)
    img = x_test[case_num].ravel()
    np.copyto(pagelocked_buffer, img)
    return y_test[case_num]


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
    # Load the shared object file containing the Clip plugin implementation.
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(MODEL_PATH),
            "Please use 'python lenet5.py' to train and save the model.",
            "For more information, see the included README.md"
        ))

    model_path = os.path.join(os.path.dirname(__file__), "models")
    # model_file = os.path.join(model_path, ModelData.MODEL_FILE)
    engine_file = os.path.join(model_path, ModelData.ENGINE_FILE)

    # Build an engine and retrieve the image mean from the model.
    with build_engine(MODEL_PATH) as engine:
        """ # 保存成engin文件，再加载会报错
        # 序列化引擎并写入文件：
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
    # 从文件中读取引擎并反序列化：
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine:
        # engine = runtime.deserialize_cuda_engine(f.read())
        """
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


if __name__ == "__main__":
    main()

"""
保存engine(成功) ,再加载报错
"""