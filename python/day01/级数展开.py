# 用展开式求指数函数的数值
"""
e^x=1+x+x^2/2! + x^3/3! + ... + x^n/n!
"""
import numpy as np

n,x=list(map(float,input("2个数:").split(" ")))
term=1.
sum=1.

for i in range(1,int(n)):
    term = term * x / i
    sum = sum + term

print(sum)
print(np.exp(x))