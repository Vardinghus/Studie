# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:40:07 2017

@author: Frederik Vardinghus
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

inter = 10
    
n = np.linspace(-inter,inter,2*inter+1)
def u(n,x):
    return (n+x >= 0)
def d(n,x):
    return (n+x == 0)
def r(n,x):
    return n*(n+x >= 0)

f = r(n,0)-2*r(n,-5)+r(n,-10)

plt.stem(n,f)









