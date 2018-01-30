# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:08:05 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

j = np.complex(0,1)
N = 500

x = np.zeros(N)
#x[2] = 1
lin = np.linspace(0,1,N)
for i in range(len(lin)/2):
#    x[i] = np.sin(lin[i])
     x[i] = 1
for i in range(len(lin)/2):
    x[i+N/2] = 0

 
def X(x,f):
    a = 0
    for n in range(len(x)):
        a += x[n]*np.exp(-j*2*np.pi*f*n)
    return a

#lin2 = np.linspace(-0.5,0.5,N)
lin2 = np.linspace(0,1,N)

amp = np.zeros(len(lin),dtype="complex64")
for i in range(len(amp)):
    amp[i] = X(x,lin2[i])
    
#plt.plot(lin,np.imag(amp))
plt.plot(lin,np.fft.fft(x))