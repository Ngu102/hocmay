import numpy as np
x = np.array([
    [60,2,10],
    [40,2,5],
    [100,3,7]
])
y=np.array([10,12,20])

N, D=x.shape
print(N,D)

bias = np.ones((N,1))
print(bias)

x_bias =np.hstack([bias,x])
print(x_bias)
print(x_bias.shape)

w=np.linalg.pinv(x_bias) @ y
print(w)

y_pred = x_bias @ w
print(y_pred)
print(y)

mse = np.mean((y-y_pred)**2)
print(mse)

x_new = np.array([1,50,2,8])
y_new=x_new @ w
print(y_new)

x_new2 = np.array([1,51,2,8])
y_new2 = x_new2 @ w
print(y_new2)
print(y_new2-y_new)

# fitting
xs = x[:, [0]]
print(xs)
print(xs.shape)

xs_bias = np.hstack([bias,xs])
print(xs_bias)
print(xs_bias.shape)
ws = np.linalg.pinv(xs_bias) @ y 
print(ws)

y_pred_s = xs_bias @ ws
print(y_pred_s)
print(y)

mse_s = np.mean((y-y_pred_s)**2)
print(mse_s)


