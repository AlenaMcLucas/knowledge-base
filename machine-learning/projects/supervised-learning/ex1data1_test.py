#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:13:41 2017

@author: alenamclucas
"""

# imports
import pandas as pd
import numpy as np
import linear_reg as linreg
import matplotlib.pyplot as plt

# load and save data
data = np.loadtxt('ex1data1.txt', delimiter=',')

X = pd.DataFrame(np.c_[data[:,0]])
y = pd.DataFrame(np.c_[data[:,1]])

(beta, costs) = linreg.gradient_descent(X, y, 0.01, 1000)
print beta
print costs

line_x = np.arange(4,24)
line_y = beta[0] + (beta[1] * line_x)

plt.scatter(pd.DataFrame(np.c_[data[:,0]]), y)
plt.plot(line_x, line_y)
plt.show()

# plot cost over time (gradient descent)
#plt.ylim(4.3, 6)
#plt.xlim(0, 1500)
#plt.plot(range(1500), costs)
#plt.show()

data2 = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])  
data2.head()  

data2 = (data2 - data2.mean()) / data2.std()   # feature normailzation
data2.head()

X_t = data2.iloc[:,:2]
y_t = pd.DataFrame(data2.iloc[:,2])

(beta, costs) = linreg.gradient_descent(X_t, y_t, 0.01, 1000)
print beta

# get cost with betas
(X_p, y_p, theta, m) = linreg.reg_prep(X_t.iloc[:,1:], y_t)
print linreg.cost_function(X_p, y_p, beta, m)
