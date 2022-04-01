#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:46:08 2022

@author: ianwallgren
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''Task 1 Write three Python functions, one which performs one step of Newton-
iteration with exact derivative, one which does the same, but with a
finite-difference approximation of the derivative and one for the secant
method. Solve with these functions the problem f (x) = arctan(x) = 0
by calling them in Python within an appropriate for-loop.'''

#f(x) = arctan(x) = 0

class Experiment:
    def __init__(self):
        pass
    
    def one_step_newton(f,Df,x0,epsilon,max_iter):
        xn = x0
        list_tracking = []
        for n in range(0,max_iter):
            fxn = f(xn)
            if abs(fxn) < epsilon:
                print('Solution found, number of iterations:',n,'.')
                return xn, list_tracking
            list_tracking.append(xn)
            Dfxn = Df(xn)
            if Dfxn == 0:
                print('Derivative is equal to zero, try another initial guess')
                return None
            xn = xn - fxn/Dfxn
        print('Increase max number of iteration.')
        return None
    
func = lambda x: np.arctan(x)
deriv = lambda x: 1/((x**2)+1)
approximate = Experiment.one_step_newton(func, deriv, 1.3, 1e-4,10)[0]
trace = Experiment.one_step_newton(func, deriv, 1, 1e-10,10)[1]

plt.plot(trace,label='tracking attempts')
plt.xlabel('number of iterations')
plt.ylabel('value')
plt.title('Example of the iterative procedure from Newton method')