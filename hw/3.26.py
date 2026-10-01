x = 5              
eta = 0.2          

def f(x):
    return x * x - 4 * x + 5

def grad(x):
    return 2 * x - 4

print("step 0: x = %.4f, f(x) = %.4f" % (x, f(x)))

for buoc in range(1, 5):
    x = x - eta * grad(x)      
    print("step %d: x = %.4f, f(x) = %.4f" % (buoc, x, f(x)))