from numpy import *
from numpy.linalg import solve
import matplotlib.pyplot as plt
# from scipy import *

###### TASK 1 ######

##  Low-effort approach using polyfit
x = [-5, -4, -3, -2, -1]
temp = [-1.9, -3.7, -5.77, 2.53, 4.32]
energy = [109.26, 92.4, 115.33, 107.77, 61.14]
PC_energy = polyfit(x, energy, 4)
PC_temp = polyfit(x, temp, 4)

temp_poly = lambda x: PC_temp[0]*x**4 + PC_temp[1]*x**3 + PC_temp[2]*x**2 + PC_temp[3]*x + PC_temp[4]
energy_poly = lambda x: PC_energy[0]*x**4 + PC_energy[1]*x**3 + PC_energy[2]*x**2 + PC_energy[3]*x + PC_energy[4]
x_grid = linspace(-5,5,5000)
temp_grid = [temp_poly(x) for x in x_grid]
energy_grid = [energy_poly(x) for x in x_grid]

plt.plot(x_grid,temp_grid)
plt.plot(x_grid,energy_grid)
plt.scatter(x,temp)
plt.scatter(x,energy)
plt.title("Energy Usage interpolation vs days")
plt.style.use("ggplot")
plt.show()

xp = 7
print(xp, temp_poly(xp))




##  Vander Approach
my_vander = vander(x,5)
PC_temp_v = solve(my_vander,temp)
PC_energy_v = solve(my_vander,energy)


### Lagrange Approach

#%%
#Lagrange polynomial
m = len(x)
n = m-1 #degree of polynomial

x = np.array([-5, -4, -3, -2, -1],float)
y = np.array([-1.9, -3.7, -5.77, 2.53, 4.32],float)
energy = np.array([109.26, 92.4, 115.33, 107.77, 61.14], float)

xplt = np.linspace(x[0],x[-1])
yplt = np.array([],float)

for xp in xplt:
    yp = 0
    for xi,yi in zip(x,y):
        yp += yi * np.prod((xp- x[x != xi])/(xi - x[x != xi]))
    yplt = np.append(yplt,yp)


plt.plot(x,y,'ro')
plt.plot(xplt,yplt,'g-')





###### TASK 2 ######

n = 15
inter = np.linspace(-1,1,n)
inter_ = sorted(np.random.uniform(-1,1,n))


wn = 0
plt_wn = []

def error(x,interval):
    wn = 1
    for i in range(len(inter)):
        wn += wn * np.prod(x-inter[i])
        plt_wn.append(wn)

    return pltwn

l = error(0.111,inter)
plt.plot(inter_,l)
plt.show()
