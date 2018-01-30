# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:07:16 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


N = 1000

x = np.zeros(N)

def ar(x,n):
    return 0.5*x[n-1] + 0.25*x[n-2] + np.random.normal(0,1)

x[0] = np.random.normal(0,1)
x[1] = 0.5*x[0] + np.random.normal(0,1)

for i in range(N-2):
    x[i+2] = ar(x,i+2)

X = np.abs(np.fft.fft(x))[:N/2]

plt.plot(X)


         
         
         
         
         
         
         
         
         
         
         
         
         