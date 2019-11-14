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
    onnx_file_path = "model.onnx"
    engine_file_path = "model.engine"
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32 #trt.float16
    NP_DTYPE = np.float32
    BATCH_SIZE = 32

def build_engine(onnx_file_path):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = ModelData.BATCH_SIZE
        builder.max_workspace_size = common.GiB(1)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as fp:
            print('Beginning ONNX file parsing')
            parser.parse(fp.read())
        print('Completed parsing of ONNX file')
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def main():
    _, _ = common.find_sample_data(description="Runs an MNIST network using a PyTorch model", subfolder="mnist")
    with build_engine(ModelData.onnx_file_path) as engine:
        # 将模型序列化为模型流：
        # serialized_engine = engine.serialize()
        # 序列化引擎并写入文件：
        with open(ModelData.engine_file_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    main()