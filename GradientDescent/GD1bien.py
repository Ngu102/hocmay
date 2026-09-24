import math
import numpy as np
import matplotlib.pyplot as plt


def grad(x):
    return x**2-1
def cost(x):
    return (1/3)*x**3-x
def myGD1(eta, x0, x_limit=50):
    x=[x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])

        if abs(x_new) > x_limit:
            x.append(x_new)
            break
        if abs(grad(x_new)) <1e-4:
            break
        x.append(x_new)
    return(x,it)

(x1, it1) = myGD1(.1, 4)
(x2, it2) = myGD1(.1, -0.5)

print(x1[-1], cost(x1[-1]), it1)
print(x2[-1], cost(x2[-1]), it2)

x_range = np.linspace(-6, 6, 100)
y_range = cost(x_range)

plt.plot(x_range, y_range)
plt.scatter(x1, [cost(xi) for xi in x1], color='red')
x2_plot = [xi for xi in x2 if abs(xi) <= 6]
plt.scatter(x2_plot, [cost(xi) for xi in x2_plot], color='green')
plt.show()