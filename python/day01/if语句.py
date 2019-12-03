n0=0;n1=0;n2=0;n3=0;n4=0
while True:
    score=float(input("score:"))
    if score<0 or score>100:
        print('满分=', n0, ' 优=', n1, ' 良=', n2, ' 及格=', n3, ' 不及格=', n4)
        exit(-1)
    else:
        if score==100:
            print('满分')
            n0 = n0 + 1
        elif score>=85 and score<=99:
            print('优  ')
            n1 = n1 + 1
        elif score >= 70 and score <=84:
            print('良  ')
            n2 = n2 + 1
        elif score >= 60 and score <=69:
            print('及格  ')
            n3 = n3 + 1
        else:
            print('不及格')
            n4 = n4 + 1