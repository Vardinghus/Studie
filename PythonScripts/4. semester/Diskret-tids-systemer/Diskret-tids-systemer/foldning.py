# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 11:05:24 2017

@author: Frederik Vardinghus
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

N = 1000 #Steps
inter = np.pi/2 #Length of interval
lin = np.linspace(0,inter,N)

x = np.zeros(N)
#for i in range(N):
#    x[i] = np.sin(lin[i])
for i in range(N):
    x[i] = np.cos(lin[i])

h = np.zeros(N)
for i in range(N):
    h[i] = np.sin(lin[i])

y = np.zeros(N)

for n in range(N):
    a = 0
    for k in range(N):
        a += x[k]*h[n-k]
    y[n]=a/N
        


plt.plot(lin,y)
plt.show()

