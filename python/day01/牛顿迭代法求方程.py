"""
# 用牛顿迭代法求方程
x^3+9.2*x^2+16.7*x+4=0 在x=0附近的根，精度满足|x_(n+1)-x_n|<=1e-5时停止迭代，并规定最多迭代50次
牛顿迭代公式为:x_(n+1)=x_n-f(x_n)/f'(x_n)
"""
def f(x:float)->float:
    return x ** 3 + 9.2 * x ** 2 + 16.7 * x + 4;
def df(x:float)->float:
    return 3 * x ** 2 + 2 * 9.2 * x + 16.7;

def newton_iteration(x:int=0,err:float=1.e-5,max_iter=50)->float:
    x_next=x-f(x)/df(x)
    iter_count=0
    while abs(x_next-x)>err and iter_count<=max_iter:
        iter_count+=1
        x=x_next
        x_next = x - f(x) / df(x)

    # print(iter_count)
    return x_next

if __name__=="__main__":
    x=newton_iteration()
    print(f(x))