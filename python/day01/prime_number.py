# 对一个大于或等于3的正整数，判断它是不是一个素数
import numpy as np
import time

def getTime(func):
    def wrapper(*arg, **kw):
        start=time.time()
        func(*arg, **kw)
        print("%s cost time:"%(str(func).split(" ")[1]),time.time()-start)
        # return func(*arg, **kw)
    return wrapper


@getTime
def is_prime_number(n:int)->int:
    j = int(np.sqrt(n))
    i=2
    while n%i!=0 and i<=j:
        i+=1

    if i<j:
        print('不是素数，可被', i, '整除')
    else:
        print( '是素数')

    return 0
@getTime
def is_prime_number2(n:int)->int:
    j = int(np.sqrt(n))
    i = 2
    k = 0
    while i<=j:
        if n%i==0:
            k=i
            break
        i+=1

    if k!=0:
        print('不是素数，可被', i, '整除')
    else:
        print('是素数')

    return 0



if __name__=="__main__":
    n = int(input("a number:"))
    is_prime_number(n)
    is_prime_number2(n)
