# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt


M = 10000

U = np.random.normal(3,9,M)

X = np.zeros(M)

for i in range(M):
    if i == 0:
        X[i] = U[i]
    else:
        X[i] = U[i] - 0.5*U[i-1]
        
Y = np.zeros(M)

for i in range(M):
    Y[i] = np.mean(X[:i])

plt.plot(Y)
plt.plot([0,M],[1.5,1.5])

















