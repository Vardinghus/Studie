# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:57:36 2017

@author: Frederik Vardinghus
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

N = 50
iters = 100


def h(x):
        return 1/8*np.exp(-x/8)

lin = np.linspace(1,N,N)

x = np.ones(N)

y = np.zeros(N) 

def y(n):
    a = 0
    for i in range(iters):
        if n-i > -1:
            a += h(i)*x[n-i]
        else:
            return a

out = np.zeros(N)

for i in range(N):
    out[i] = y(i)
    
plt.plot(lin,out)
    
    









