import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

# sinh data
N=10
cov=[[0.3, 0.2], [0.2, 0.3]]
x0=np.random.multivariate_normal([2,2], cov, N) #nhom1
x1=np.random.multivariate_normal([4,2], cov, N) #nhom2

X=np.vstack([x0,x1]) #(20, 2)
y=np.hstack([-np.ones(N), np.ones(N)]) #(20, )
X=np.hstack([np.ones((2*N,1)), X]) # gộp bias (20, 3)

# diem nao dang sai
def misclassified(w,X,y):
    return np.where(y*(X@w) <= 0)[0] # chi so cac diem sai

# vong lap pla, luu ls
def pla(X,y):
    w=np.zeros(X.shape[1])
    history = [] # mỗi (w, i)
    while True:
        wrong = misclassified(w,X,y)
        if len(wrong) == 0: 
            history.append((w.copy(), None))
            break
        i = np.random.choice(wrong) # diem se duoc khoang tron
        history.append((w.copy(), i)) # luu truoc khi update
        w = w + y[i]*X[i]
    return w,history

     
def draw(k, X, y, history):
    w, i = history[k]
    print(k,i)
    plt.figure(figsize=(5, 5))
    plt.plot(X[y == -1, 1], X[y == -1, 2], 'b^')       # tam giác xanh
    plt.plot(X[y ==  1, 1], X[y ==  1, 2], 'ro')       # chấm đỏ
    

    if i is not None:
        plt.plot(X[i, 1], X[i, 2], 'o', ms=15, mfc='none', mec='k')

    xs = np.array([X[:, 1].min() - 1, X[:, 1].max() + 1])
    if abs(w[2]) > 1e-12:
        plt.plot(xs, -(w[0] + w[1] * xs) / w[2], 'k-')  # w0 + w1 x + w2 y = 0
    plt.ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)
    plt.xlabel(f"PLA: iter {k+1}/{len(history)}")
    plt.show()

w, history = pla(X, y)
for k in range(len(history)):
    draw(k, X, y, history)