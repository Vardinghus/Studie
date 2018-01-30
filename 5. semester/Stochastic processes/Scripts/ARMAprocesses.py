# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt


N = 100

x = np.linspace(0,N,N)

phi1 = 0.9
phi2 = -0.5
sigma = 1

start = np.array([[-1,phi1,phi2 ],
                  [phi1,phi2-1,0],
                  [phi2,phi1,-1 ]])

vec = np.array([[-sigma],
                [0],
                [0]])

sol = np.dot(np.linalg.inv(start),vec)

r = np.zeros(N)
r[0] = sol[0]
r[1] = sol[1]
r[2] = sol[2]

x = np.linspace(0,N-1,N)
    
def acf(r,k):
    if k == 0:
        return phi1*r[np.abs(k-1)] + phi2*r[np.abs(k-2)] + sigma
    else:
        return phi1*r[np.abs(k-1)] + phi2*r[np.abs(k-2)]

for i in range(N-3):
   r[i+3] = acf(r,x[i+3])

S = np.zeros(N)
f = np.linspace(-0.5,0.5,N)
for i in range(N):
    S[i] = sigma/(np.abs(1-phi1*np.exp(-2*np.pi*1j*f[i])-phi2*np.exp(-2*np.pi*1j*f[i])))**2

plt.plot(r)

















    
    
    
    
    
    
    
    