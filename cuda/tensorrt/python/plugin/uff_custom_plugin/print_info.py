import numpy as np

weights=np.load("models/tf_args.npz")

for k,v in weights.items():
    print("%s:\t%s"%(k,v.shape))

"""
OutputLayer/bias:0:	(10,)
dense/bias:0:	(512,)
OutputLayer/kernel:0:	(512, 10)
dense/kernel:0:	(784, 512)
"""