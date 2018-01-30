# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:57:26 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt

order = 100
steps = 1000
start = -2*np.pi
stop = 2*np.pi

def f(N,x):
    a = 0
    for n in range(N):
        a += (n+1)**(-1)*np.sin((n+1)*x)
    a = 2*a
    return a

lin = np.linspace(start, stop, steps)

sol = np.zeros(steps)

for i in range(steps):
    sol[i] = f(order,lin[i])

plt.plot(lin,sol)
