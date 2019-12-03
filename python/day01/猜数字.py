import numpy as np

Number=np.random.randint(1,10,1)[0]
n = 0
while True:
    MyGuess=int(input("Your Guess:"))
    if MyGuess == Number: break
    if MyGuess > Number:
        print('Too high. Try again.')
    else:
        print('Too low.  Try again.')

print('You are lucky. It is ',Number)
