# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.zeros(N)

def AR(x,n):
    return 0.50*x[n] + 0.25*x[n-1] + np.random.normal(0,1)

def bacf(x,k):
    M = len(x)
    if np.abs(k) < M:
        a = 0
        for i in range(M-np.abs(k)):
            a += x[i]*x[i-k]
        return a/float(M)
    else:
        return 0

def ubacf(x,k):
    M = len(x)
    if np.abs(k) < M:
        a = 0
        for i in range(M-np.abs(k)):
            a += x[i]*x[i-k]
        return a/float(M-np.abs(k))
    else:
        return 0

x[0] = np.random.normal(0,1)
x[1] = 0.5*x[0] + np.random.normal(0,1)

for i in range(N-1):
    x[i+1] = AR(x,i)

rb = np.zeros(N)
ru = np.zeros(N)
for i in range(N):
    rb[i] = bacf(x,i)
    ru[i] = ubacf(x,i)

plt.plot(rb)
plt.plot(ru)



















