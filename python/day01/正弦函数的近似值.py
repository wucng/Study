# 正弦函数的近似值
# !正弦函数用泰勒级数展开：sinx=x-x^3/3!+x^5/5!-x^7/7!+...。计算有限精度范围内的值。

import numpy as np

def sin_approx(x:float,err:float=1.e-6,max_terms:int=10)->float:
    k=1; term=x; sin_=term
    while abs(term)>err and k<=max_terms:
        term=-term*x*x/(2*k*(2*k+1))
        k = k + 1
        sin_ = sin_ + term
        # print(k, sin_)
    return sin_

if __name__=="__main__":
    x = float(input("a float number:"))
    # pi=np.pi
    pi = np.arccos(-1)
    x = x * pi / 180
    err = 1.e-6
    max_terms = 10
    print(sin_approx(x))
    print(np.sin(x))
