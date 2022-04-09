import numpy as np
import matplotlib.pyplot as plt
from truck import Truck
from numpy import cos, arctan

### Task 1
truck = Truck()
f1 = truck.fcn
x0 = truck.initial_conditions()

# Define Jacobian
def jacobian(func, t, x, eps=1e-8):
    n = len(x)
    J = np.zeros((n, n))
    fx = func(t, x)
    
    for i in range(n):
        x[i] += eps
        J[:, i] = (func(t, x) - fx)/eps
        x[i] -= eps
    
    return J
    
# Newton iteration in R^n
def newton(func, t, x0):
    jac = jacobian(func, t, x0)
    delx = np.linalg.solve(jac, -1 * func(t, x0))
    return x0 + delx


# Main iteration
t = 0
for i in range(50):
    norm = np.linalg.norm(x0)
    x0 = newton(f1, t, x0)
    
#%%
### Task 2

# Constants
k, h, l = 10, 2, 1

# Left-hand side function
def func(x, F):
    term1 = 2*k*(h-x)/l * cos(arctan(h/l))
    term2 = -cos(arctan((h-x)/l)) + F
    return term1 + term2

# Finite newton: Too lazy to compute derivate
def finite_newton(f, x, F, eps=1e-4):
    return x - f(x, F)*2*eps / (f(x+eps, F) - f(x-eps, F))

# Uses newton to iterate to convergence for given F
def solver(x0, F):
    x1 = x0 - 1
    while (abs(x1 - x0) > 1e-13):
        x1 = x0
        x0 = finite_newton(func, x0, F)
    return x0

# Solve for each F in [0, 20]
F = np.linspace(0, 20, 40)
x = [solver(3, f) for f in F]

# Plotting
plt.scatter(F, x, marker="x")
plt.xlabel("Force [N]")
plt.ylabel("Displacement [m]")
plt.grid()
