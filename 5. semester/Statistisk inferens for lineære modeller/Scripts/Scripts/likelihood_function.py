# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 14:49:56 2017

@author: Frederik Vardinghus
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt

rep = 10
success = 3

def L(theta):
    return m.factorial(rep)/(m.factorial(success)*m.factorial((rep-success)))*(theta**success)*(1-theta)**(rep-success)

x = np.linspace(0,1,100)
y = np.zeros(100)

for i in range(len(x)):
    y[i] = L(x[i])

plt.plot(x,y)











