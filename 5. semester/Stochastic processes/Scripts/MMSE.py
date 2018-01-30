# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10000

T = np.random.uniform(0,3,N)
W = np.random.uniform(0,1,N)
Z = T + W

def MMSE(z):
    if 0 <= z < 1:
        return 0.5*z
    elif 1 <= z <= 3:
        return z-0.5
    elif 3 < z <= 4:
        return 0.5*z+1

hat = np.zeros(N)
for i in range(N):
    hat[i] = MMSE(Z[i])

bias = 0
MSE = 0
for i in range(N):
    bias += T[i] - hat[i]
    MSE += (T[i] - hat[i])**2
bias = bias/float(N)
MSE = MSE/float(N)



#==============================================================================
# LMMSE
#==============================================================================

Et = 1.5
Ez = 2
Varz = (3**2)/12. + 1/12.
Cov = 3/4.

h = Varz**(-1)*Cov
h0 = Et - h*Ez

def lmmse(z):
    return -3/10. + 9/10.*z

bias_l = T - lmmse(Z)
MSE_l = (T - lmmse(Z))**2
bias_l = np.sum(bias_l)/float(N)
MSE_l = np.sum(MSE_l)/float(N)

print('Bias af MMSE: %f' %bias)
print('MSE af MMSE: %f' %MSE)
print('')
print('Bias af LMMSE: %f' %bias_l)
print('MSE af LMMSE: %f' %MSE_l)












