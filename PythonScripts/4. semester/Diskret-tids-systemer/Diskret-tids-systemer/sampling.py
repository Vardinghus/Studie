# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:30:02 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
n = 9


lin = np.linspace(0,2*np.pi,N)

x = np.zeros(len(lin))
for i in range(len(lin)):
    x[i]=np.sin(lin[i])

xs = np.zeros(n)
for i in range(n):
    xs[i]=x[i*N/n]

y = np.zeros(len(lin))
for i in range(len(lin)):
    y[i]=np.sin(10*lin[i])
    
ys = np.zeros(n)
for i in range(n):
    ys[i]=y[i*N/n]
    
lin2=np.linspace(-np.pi,np.pi,len(lin))
lin3=np.linspace(0,2*np.pi,n)

plt.plot(xs)
plt.plot(ys)
#plt.plot(lin2,y)

#plt.plot(np.fft.fft(y))