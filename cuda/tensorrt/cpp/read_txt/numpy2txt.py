"""
先将numpy array保存成txt文件(格式比较自由)，再使用C++ 加载数据
"""
import json
import numpy as np
"""
jdata={}

jdata["dense/w1"]={
    "rows": 3,
    "cols":6,
    "data":np.random.randn(3,6).ravel().tolist()
}

jdata["dense/b1"]={
    "rows": 6,
    "cols":1,
    "data":np.random.randn(6).ravel().tolist()
}

# json.dump(jdata,open("test.json",'w', encoding='utf-8'),ensure_ascii=False,indent=4)
"""

with open("test.txt","w") as fp:
    # fp.write(json.dumps(jdata))
    fp.write("dense/w1"+"\n")
    fp.write("rows:" + str(3) +"\n")
    fp.write("cols:" + str(6) +"\n")
    fp.write("data:" + str(np.random.randn(3,6).ravel().tolist()) +"\n")
    fp.write("\n")
    fp.write("dense/b1" + "\n")
    fp.write("rows:" +str(6)+"\n")
    fp.write("cols:" +str(1)+"\n")
    fp.write("data:" +str(np.random.randn(6).ravel().tolist())+"\n")
