[toc]

# 1、模型训练
```python
# 1、模型训练，生成.pb文件(冻结网络参数)
python3 model.py
```
冻结网络参数得到文件`models/lenet5.pb`

将权重转成`numpy array`保存成`tf_args.npz`

# 2、.pb文件转成.uff文件
```python
# 2、.pb文件转成.uff
convert-to-uff models/lenet5.pb

# or 查看 xxx/uff/bin/convert_to_uff.py文件改写成脚本
cp /usr/lib/python3.5/site-packages/uff/bin/convert_to_uff.py ./
python3 convert_to_uff.py models/lenet5.pb
```

# 3、推理计算
## 3.1、numpy直接加载npz文件做推理
```python
python3 numpy_load_npz_do_classify.py
```

## 3.2、tensorrt加载.uff文件实现模型推理
```python
python3 load_uff_do_classify.py
```

## 3.3 将uff序列化，在加载做推理 （推荐）
```python
# 1、将uff文件直接序列化为engine文件(tensorrt文件)
python3 uff_to_engine.py

# 2、直接加载engine文件做推理
python3 load_engine_do_classify.py
```

## 3.4 tensorrt加载npz文件序列化做做推理
```python
python3 npz_to_engine.py

python3 load_engine_do_classify.py
```
# 注意
```python
# tensorflow 输入格式为[N,H,W,C],而tensorrt(pytorch)要求的格式为[N,C,H,W],
# tensorflow 权重格式[f,f,in,out] ,tensorrt(pytorch)权重格式：[out,in,f,f]
# fc1_w = weights['fc1.weight'] # 从pytorch加载权重,不需要转置，可以不加reshape(-1)
# fc1_b = weights['fc1.bias']
# fc1_w = weights['dense/kernel:0'].tanspose([1,0]) #.transpose((3,2,0,1))
fc1_w = np.transpose(weights['dense/kernel:0'],[1,0]).reshape(-1) # 从tensorflow加载权重必须先转置，最后必须加上reshape(-1)，否则结果不对
```



