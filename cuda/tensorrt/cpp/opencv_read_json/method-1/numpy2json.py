"""
先将numpy array保存成json文件(格式比较自由)，再使用C++ opencv加载数据
"""
import json
import numpy as np

jdata={}

jdata["dense/w1"]={
    "in_channel": 3,
    "out_channel":6,
    "value":np.random.randn(3,6).ravel().tolist()
}

jdata["dense/b1"]={
    "out_channel": 6,
    "value":np.random.randn(6).ravel().tolist()
}

json.dump(jdata,open("test.json",'w', encoding='utf-8'),ensure_ascii=False,indent=4)


