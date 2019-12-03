# 求两个整数的最大公约数。
# !碾转相除法：如27和6，先用27除以6，余数为3。再用6除以3，余数为零，即3为最大公约数。
import time

def getTime(func):
    def wrapper(*arg, **kw):
        start=time.time()
        func(*arg, **kw)
        print("%s cost time:"%(str(func).split(" ")[1]),time.time()-start)
        # return func(*arg, **kw)
    return wrapper

@getTime
def common_divisor(m:int,n:int)->None:
    if m<n:
        k=m
        m=n
        n=k

    k=m%n
    while k!=0:
        m=n;n=k;k=m%n

    print('最大公约数＝',n)

@getTime
def common_divisor2(m:int,n:int)->None:
    while m!=n:
        while m>n:
            m-=n
        while n>m:
            n-=m

    print('最大公约数＝', m)

if __name__=="__main__":
    m = int(input("number1:"))
    n = int(input("number2:"))
    common_divisor(m,n)
    common_divisor2(m,n)

