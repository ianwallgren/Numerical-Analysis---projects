from numpy import *
from numpy.linalg import solve
import matplotlib.pyplot as plt
# from scipy import *


##  Low-effort approach using polyfit
x = [-1.9, -3.7, -5.77, 2.53, 4.32]
y = [109.26, 92.4, 115.33, 107.77, 61.14]
PC = polyfit(x,y,3)

my_poly = lambda x: PC[0]*x**3 + PC[1]*x**2 + PC[2]*x + PC[3]
x_grid = linspace(-10,10,5000)
y_grid = [my_poly(x) for x in x_grid]

plt.plot(x_grid,y_grid)
plt.scatter(x,y)
plt.title("Energy Usage interpolation vs Temperature")
plt.style.use("ggplot")
plt.show()



##  Vander Approach
my_vander = vander(x,)
yt = transpose(y)

my_coeffs = solve(my_vander,yt)
