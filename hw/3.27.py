import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])   
y = -1

s = w @ x 
print(s)                     
y_pred = 1 if s >= 0 else -1      
sai = (y * s <= 0)                         
print(y_pred)  
print(sai)      