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

    target_list=[]
    pred_list=[]
    with engine.create_execution_context() as context:
        for data, target in test_loader:
            data = data.numpy() # data.numpy().ravel().astype(np.float32)  # 展成一行
            case_num = target.numpy()
            target_list.extend(case_num)

            # 不足一个batch填充0,补齐
            len_data=len(case_num)
            if len_data!=ModelData.BATCH_SIZE:
                tmp_data=np.zeros([ModelData.BATCH_SIZE,*ModelData.INPUT_SHAPE])
                tmp_target=np.zeros([ModelData.BATCH_SIZE],dtype=np.int8)
                tmp_data[:len_data]=data
                tmp_target[:len_data]=case_num
                data=tmp_data
                case_num=tmp_target

            data = data.ravel().astype(ModelData.NP_DTYPE)  # 展成一行
            np.copyto(inputs[0].host, data)

            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, \
                                           inputs=inputs, outputs=outputs, stream=stream, \
                                           batch_size=ModelData.BATCH_SIZE)
            output = np.reshape(output, [-1, ModelData.OUTPUT_SIZE])  # 转成[-1,10]
            pred = np.argmax(output, -1)
            # print("Test Case: " + str(case_num[:len_data]))
            # print("Prediction: " + str(pred[:len_data]))
            pred_list.extend(pred[:len_data])

        # 计算测试精度
        nums=len(np.asarray(pred_list))
        nums_right=np.sum(np.asarray(pred_list)==np.asarray(target_list))
        print("test acc:(%d / %d) = %.3f"%(nums_right,nums,nums_right/nums))

if __name__ == '__main__':
    main()

"""test acc:(1742 / 4585) = 0.380"""