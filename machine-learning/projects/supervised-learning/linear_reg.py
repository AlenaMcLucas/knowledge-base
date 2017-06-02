#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 21:58:18 2017

@author: alenamclucas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reg_prep(X, y):
    m = len(X)   # number of training examples
    theta = np.array([0.0] * (len(X.columns) + 1))   # initialize thetas
    X.insert(0, 'b0', 1.0)   # append 1s for B0
    X_a = np.array(X)   # convert to arrays
    y_a = np.array(y)
    
    return (X_a, y_a, theta, m)

def cost_function(X, y, theta, m):
    return np.sum(np.power((X.dot(theta) - y.flatten()), 2)) / (2.0 * m)

def gradient_descent(X, y, alpha, iterations):
    
    (X, y, theta, m) = reg_prep(X, y)
    
    costs = [0] * iterations   # initialze all costs

    for i in range(iterations):
        theta = theta - (float(alpha) / m) * (X.T.dot(X.dot(theta) - y.flatten()))
        costs[i] = cost_function(X, y, theta, m)   # save cost from each iteration
    
    return theta, costs