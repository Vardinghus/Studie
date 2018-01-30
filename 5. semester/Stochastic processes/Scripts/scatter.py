# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000

X = np.random.uniform(0,3,N)
Y = np.random.uniform(0,1,N)
Z = X + Y

plt.plot(X,Y,'.')
plt.plot(Z,X,'.')

