# import numpy as np

s=0;k=0;n=0

while True:
    n=int(input('键入分数 （负数时退出）：'))
    if n<0:
        print(k,' 个学生合格。 总人数＝', s)
        break
    elif n>=60:
        k+=1
    else:
        pass
    s = s + 1
