# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

start = time.time()

N = 100

h = 0.95
sigmaZ = 0.1
sigmaW = 1
sigmaY = sigmaZ/(1-h)

y = np.zeros(N)
y[0] = np.random.normal(0,sigmaY)
x = np.zeros(N)

#==============================================================================
# LMMSEE
#==============================================================================

for i in range(N-1):
    y[i+1] = h*y[i] + np.random.normal(0,sigmaZ)

for i in range(N):
    x[i] = y[i] + np.random.normal(0,sigmaW)

def ry(k):
    return sigmaZ*(h**(np.abs(k))/(1-h**2))

def rx(k):
    if k == 0:
        return ry(k) + sigmaW
    else:
        return ry(k)

r = np.zeros(N)
for i in range(N):
    r[i] = rx(i)

sigmaxx = sp.linalg.toeplitz(r)

sigmaxy = np.zeros((N,1))
for i in range(N):
    sigmaxy[i] = ry(i)

hbar = np.dot(np.linalg.inv(sigmaxx),sigmaxy)

LMMSEE = np.zeros(N)
for i in range(N):
    for j in range(i):
        LMMSEE[i] += hbar[j]*x[i-j]

plt.figure(1)
plt.plot(y)
plt.plot(LMMSEE)    

#==============================================================================
# Kalmanfilter
#==============================================================================

ypred = np.zeros(N)
xpred = np.zeros(N)
Rpred = np.zeros(N)
b = np.zeros(N)

R = np.zeros(N)
R[0] = sigmaW
yhat = np.zeros(N)
yhat[0] = 0

for i in range(1,N):
    ypred[i] = h*yhat[i-1]
    xpred[i] = ypred[i]
    Rpred[i] = (h**2)*R[i-1] + sigmaZ
    b[i] = Rpred[i]/(Rpred[i] + sigmaW)
    R[i] = (1-b[i])*Rpred[i]
    yhat[i] = ypred[i] + b[i]*(x[i] - xpred[i])

plt.plot(yhat,alpha=0.6)

plt.figure(2)
plt.plot(R)

#==============================================================================
# Beregn steady state for Kalmanfilteret
#==============================================================================

a = h**2
b = sigmaZ + sigmaW -sigmaW*h**2
c = -0.1

d = b**2 - 4*a*c
x1 = (-b-np.sqrt(d))/(2*a)
x2 = (-b+np.sqrt(d))/(2*a)    






