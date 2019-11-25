"""
参考:https://docs.opencv.org/master/dd/d74/tutorial_file_input_output_with_xml_yml.html
先将numpy array保存成json文件(按照opencv保存mat的格式保存)，再使用C++ opencv加载数据

fc层(dense层)的权重推荐使用这种方式，conv层的权重使用方式一解析
"""
import json
import numpy as np

jdata={}

# "dt":"u"  #uchar
# "dt":"d"  #double
# "dt":"f"  #float
jdata["dense/w1"]={
    "type_id": "opencv-matrix",
    "rows": 3,
    "cols":6,
    "dt": "f",
    "data":np.random.randn(3,6).astype("f").ravel().tolist()
}

jdata["dense/b1"]={
    "type_id": "opencv-matrix",
    "rows": 6,
    "cols": 1,
    "dt": "f",
    "data":np.random.randn(6).astype("f").ravel().tolist()
}

json.dump(jdata,open("test.json",'w', encoding='utf-8'),ensure_ascii=False,indent=4)


