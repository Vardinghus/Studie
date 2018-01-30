# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:49:20 2017

@author: Frederik Vardinghus
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

datafile = open('signal_noise', 'r')
data = [float(w) for w in datafile.read().split()]
data = np.array(data)
datafile.close()

N = 1000

out = np.zeros(N)

def y(n):
    if i > 0:
        return np.exp(-1/8)*out[n-1]+1/8*data[n]
    else:
        return 1/8*data[n]

for i in range(N):
    out[i] = y(i)

lin = np.linspace(0,N,N)

plt.plot(lin,data)
plt.plot(lin,out)


