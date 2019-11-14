from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
import common

from data_prepare import test_loader

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

def get_batch_testcase():
    data, target = next(iter(test_loader))
    test_case = data.numpy().ravel().astype(np.float32)  # 展成一行
    test_name = target.numpy()
    return test_case, test_name

# Loads a random test case from pytorch's DataLoader
def load_random_test_case(pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = get_batch_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


def main():
    engine_file = "model.engine"
    # 从文件中读取引擎并反序列化：
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    with engine.create_execution_context() as context:
        case_num = load_random_test_case(pagelocked_buffer=inputs[0].host)
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        [output] = common.do_inference(context, bindings=bindings, \
                                       inputs=inputs, outputs=outputs, stream=stream, \
                                       batch_size=ModelData.BATCH_SIZE)
        output = np.reshape(output, [-1, ModelData.OUTPUT_SIZE])  # 转成[-1,10]
        pred = np.argmax(output, -1)
        print("Test Case: " + str(case_num))
        print("Prediction: " + str(pred))


if __name__ == '__main__':
    main()