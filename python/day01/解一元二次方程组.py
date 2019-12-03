import numpy as np

# 求解二次方程 a*x**2+b*x+c=0

a_in = input("输入3个系数: (空格隔开)")
a,b,c=list(map(float,a_in.split(" ")))
esp =1e-10
if abs(a) <= esp:
    print('a的值小于允许的最小实数，认为是零')
    if abs(b) <= esp:
        print('b的值小于允许的最小实数，认为是零')
        if abs(c) <= esp:
            print('c的值小于允许的最小实数，认为是零')
            print('恒等式0=0，无需解')
        else:
            print('无解')
    else:
        print('一个解：x=', -c / b)

else:
    p1 = -b / (2 * a)
    s = b ** 2 - 4 * a * c
    p2 = np.sqrt(abs(s)) / (2 * a)
    if s<0:
        print('两个复数解：x=', p1, ' +-', p2, 'i')
    elif abs(s)<esp:
        print('重根      ：x=', p1)
    else:
        print('两个实数解：x=', p1, ' +-', p2)
