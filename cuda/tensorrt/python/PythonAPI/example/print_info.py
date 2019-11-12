import numpy as np

weights=np.load("model.npz")

print(list(weights.keys()))

"""
'batch_norm1.weight', 'batch_norm1.running_mean'
'batch_norm1.running_var','batch_norm1.bias'
'batch_norm1.num_batches_tracked'
"""
print(weights['batch_norm1.weight'].shape)
print(weights['batch_norm1.bias'].shape)
print(weights['batch_norm1.running_mean'].shape)
print(weights['batch_norm1.running_var'].shape)
print(weights['batch_norm1.num_batches_tracked'].shape)


"""
BN=weight*(x-mean)/var+bias
==>
BN=[(weight/var)*x+(-weight*mean/var+bias)]^1
==>
BN=(x*scale+shift)^power
"""
