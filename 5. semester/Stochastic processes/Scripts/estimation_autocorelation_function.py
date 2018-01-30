# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
h = 1000

def x(n):
    U = np.random.normal(0,1,N)
    return U[n]-0.5*U[n-1]+0.25*U[n-2]

X = np.zeros(N)
for i in range(N):
    X[i] = x(i)

theta = np.random.uniform(-np.pi,np.pi)
Y = np.zeros(N)
for i in range(N):
    Y[i] = np.cos(i+theta)

def acor(X,k):
    a = 0
    for i in range(N-np.abs(k)):
        a += X[i]*X[i+k]
    return 1/float(N-np.abs(k))*a

auto = np.zeros(h)
for i in range(h):
    auto[i] = acor(X,i)

plt.plot(np.linspace(0,h,h),auto)
print(np.mean(auto))

#plt.plot(Y)

#print(np.mean(X))






