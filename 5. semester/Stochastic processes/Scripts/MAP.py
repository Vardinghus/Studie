# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:07:16 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
k = 0
p = 0.4
a = -1
b = 1
m = 0
v = 1

x = np.linspace(-3,3,N)

def pdfu(x,a,b):
    if a <= x <= b:
        return (1/float(b-a))*p
    else:
        return 0
    
u = np.zeros(N)
for i in range(N):
    u[i] = pdfu(x[i],a,b)

def pdfn(x,m,v):
    return 1/np.sqrt(2*np.pi*(v**2))*np.exp(-(x-m)**2/float(2*v**2))*(1-p)

z = pdfn(x,m,v)

plt.plot(x,u)
plt.plot(x,z)
plt.plot(0.599,p*0.5,'rs')
plt.plot(-0.599,p*0.5,'rs')
plt.plot(1,pdfn(1,m,v),'rs')
plt.plot(-1,pdfn(-1,m,v),'rs')
















