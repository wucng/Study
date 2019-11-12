#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
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
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float16 #trt.float32
    BATCH_SIZE = 32

def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # ---------------------------------------------------------------------------------------
    conv1_w = weights['conv1.weight']
    conv1_b = weights['conv1.bias']
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w,
                                    bias=conv1_b)
    conv1.stride = (1, 1)

    g0=weights['batch_norm1.weight']#.reshape(-1)
    b0=weights['batch_norm1.bias']#.reshape(-1)
    m0=weights['batch_norm1.running_mean']#.reshape(-1)
    v0=weights['batch_norm1.running_var']#.reshape(-1)
    scale0=g0/np.sqrt(v0+1e-3)
    shift0=-m0/np.sqrt(v0+1e-3)*g0+b0
    power0=np.ones(len(g0),dtype=np.float32)
    bn1 = network.add_scale(input=conv1.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=shift0,scale=scale0,power=power0)

    relu1 = network.add_activation(input=bn1.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling(input=relu1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    # ---------------------------------------------------------------------------------------
    conv2_w = weights['conv2.weight']
    conv2_b = weights['conv2.bias']
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    g0 = weights['batch_norm2.weight']  # .reshape(-1)
    b0 = weights['batch_norm2.bias']  # .reshape(-1)
    m0 = weights['batch_norm2.running_mean']  # .reshape(-1)
    v0 = weights['batch_norm2.running_var']  # .reshape(-1)
    scale0 = g0 / np.sqrt(v0 + 1e-3)
    shift0 = -m0 / np.sqrt(v0 + 1e-3) * g0 + b0
    power0 = np.ones(len(g0), dtype=np.float32)
    bn2 = network.add_scale(input=conv2.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0,
                            power=power0)


    relu2 = network.add_activation(input=bn2.get_output(0), type=trt.ActivationType.RELU)

    pool2 = network.add_pooling(relu2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    # dropout层，viwe，reshape 都没涉及到权重信息可以不用搭建
    # ---------------------------------------------------------------------------------------
    fc1_w = weights['fc1.weight']
    fc1_b = weights['fc1.bias']
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu3 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    # ---------------------------------------------------------------------------------------
    fc2_w = weights['fc2.weight']
    fc2_b = weights['fc2.bias']
    fc2 = network.add_fully_connected(relu3.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    """
    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))
    """
    # or add softmax layer
    softmax=network.add_softmax(fc2.get_output(0))
    softmax.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax.get_output(0))
    # """


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_batch_size = ModelData.BATCH_SIZE
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    # img, expected_output = model.get_random_testcase()
    img, expected_output = model.get_batch_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output

def main():
    _, _ = common.find_sample_data(description="Runs an MNIST network using a PyTorch model", subfolder="mnist")
    # Train the PyTorch model
    mnist_model = model.MnistModel()
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

    # 从文件中读取引擎并反序列化：
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
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
