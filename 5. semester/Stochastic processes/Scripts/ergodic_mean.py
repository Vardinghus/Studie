# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

M = 1000
N = 1000

Y = np.zeros([N,M])

for i in range(N):
    A = np.random.uniform(0,3)
    W = np.random.normal(0,1,M)
    Y[i] = A+W

avg = np.zeros(N)
for i in range(N):
    avg[i] = np.mean(Y[:i])

plt.plot(avg)
plt.plot([0,N],[1.5,1.5])











