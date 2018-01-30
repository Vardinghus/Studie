# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def fun(x1,x2):
    return x1**3 + x2**3

A = np.array([[4,2],
              [2,2]])
b = np.array([[-1],
              [1]])

d = 1/(np.dot(A,x)-b)
hess = np.dot(np.dot(np.transpose(A),np.diag(np.diag(d)**2)),A)
grad =






    
    
    
    
    
    