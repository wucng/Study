from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"
    BATCH_SIZE = 32
    NUM_classes =10
    # DTYPE = trt.float32 # 设置数据精度

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_batch_size=ModelData.BATCH_SIZE # batch size
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case2(data_paths, pagelocked_buffer, case_num=randint(0, 9)):
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num

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
    data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)

    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            # case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
            case_num = load_normalized_test_case(pagelocked_buffer=inputs[0].host)

            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs,
                                         outputs=outputs, stream=stream,batch_size=ModelData.BATCH_SIZE)
            output=np.reshape(output,[-1,ModelData.NUM_classes]) # 默认是一行的，展成原来的形状
            pred = np.argmax(output,-1)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))

if __name__ == '__main__':
    main()
