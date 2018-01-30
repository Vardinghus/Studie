# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt

N = 1000

X = np.linspace(1,N,N)
Y = np.zeros(N)

for i in range(N):
    Y[i] = np.sum((np.random.uniform(-1,1,i)**3))*1/(i+1)


plt.plot(X,Y)