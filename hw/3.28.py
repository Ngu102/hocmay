import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])  
y = 1

s = w @ x
sai = (y * s <= 0)
print(s)  
print(sai)          

if sai:
    w = w + y * x
print(w)                  
print(w @ x) 