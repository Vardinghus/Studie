# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

h = 0.5

N = 10000

C = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        C[i][j] = h**(np.abs(j-i))


chol = np.linalg.cholesky(C)

X = np.random.normal(0,1,N)

sim = np.dot(chol,X)

plt.hist(sim)



