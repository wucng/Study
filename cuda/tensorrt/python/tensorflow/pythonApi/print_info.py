import numpy as np

weights=np.load("models/tf_args.npz")

for k,v in weights.items():
    print("%s:\t%s"%(k,v.shape))

"""
conv2d/kernel:0:	(5, 5, 1, 20)
conv2d/bias:0:	(20,)
batch_normalization/gamma:0:	(20,)
batch_normalization/beta:0:	(20,)

conv2d_1/kernel:0:	(5, 5, 20, 50)
conv2d_1/bias:0:	(50,)
batch_normalization_1/gamma:0:	(50,)
batch_normalization_1/beta:0:	(50,)

dense/kernel:0:	(800, 500)
dense/bias:0:	(500,)

dense_1/kernel:0:	(500, 10)
dense_1/bias:0:	(10,)
"""

"""
conv2d/bias:0:	(20,)
training/Adam/conv2d_1/bias/m:0:	(50,)
batch_normalization/moving_mean:0:	(20,)
training/Adam/dense_1/bias/m:0:	(10,)
training/Adam/beta_2:0:	()
batch_normalization_1/moving_mean:0:	(50,)
batch_normalization_1/beta:0:	(50,)
training/Adam/beta_1:0:	()
training/Adam/batch_normalization_1/beta/v:0:	(50,)
training/Adam/dense/bias/v:0:	(500,)
training/Adam/conv2d/kernel/m:0:	(5, 5, 1, 20)
batch_normalization_1/gamma:0:	(50,)
training/Adam/decay:0:	()
training/Adam/batch_normalization/gamma/m:0:	(20,)
training/Adam/conv2d_1/kernel/v:0:	(5, 5, 20, 50)
conv2d/kernel:0:	(5, 5, 1, 20)
dense/kernel:0:	(800, 500)
training/Adam/conv2d/bias/v:0:	(20,)
conv2d_1/bias:0:	(50,)
training/Adam/conv2d/bias/m:0:	(20,)
training/Adam/batch_normalization_1/gamma/v:0:	(50,)
training/Adam/dense/kernel/m:0:	(800, 500)
batch_normalization/moving_variance:0:	(20,)
training/Adam/batch_normalization/beta/m:0:	(20,)
training/Adam/dense/kernel/v:0:	(800, 500)
training/Adam/batch_normalization/gamma/v:0:	(20,)
training/Adam/dense_1/bias/v:0:	(10,)
dense_1/bias:0:	(10,)
training/Adam/dense_1/kernel/m:0:	(500, 10)
training/Adam/conv2d_1/kernel/m:0:	(5, 5, 20, 50)
training/Adam/batch_normalization_1/gamma/m:0:	(50,)
training/Adam/conv2d_1/bias/v:0:	(50,)
dense_1/kernel:0:	(500, 10)
training/Adam/batch_normalization/beta/v:0:	(20,)
training/Adam/iter:0:	()
training/Adam/learning_rate:0:	()
training/Adam/dense_1/kernel/v:0:	(500, 10)
training/Adam/batch_normalization_1/beta/m:0:	(50,)
batch_normalization/gamma:0:	(20,)
training/Adam/conv2d/kernel/v:0:	(5, 5, 1, 20)
batch_normalization_1/moving_variance:0:	(50,)
training/Adam/dense/bias/m:0:	(500,)
batch_normalization/beta:0:	(20,)
dense/bias:0:	(500,)
conv2d_1/kernel:0:	(5, 5, 20, 50)
"""

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 24, 24, 20)        520       
_________________________________________________________________
batch_normalization (BatchNo (None, 24, 24, 20)        80        
_________________________________________________________________
re_lu (ReLU)                 (None, 24, 24, 20)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 20)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 50)          25050     
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 50)          200       
_________________________________________________________________
re_lu_1 (ReLU)               (None, 8, 8, 50)          0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 50)          0         
_________________________________________________________________
flatten (Flatten)            (None, 800)               0         
_________________________________________________________________
dense (Dense)                (None, 500)               400500    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5010      
=================================================================
Total params: 431,360
Trainable params: 431,220
Non-trainable params: 140
_________________________________________________________________
"""