# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 12:34:40 2017

@author: Jonas
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def arp(x,a,n,peri,G):
    temp = 0
    if n == 0:
        return G
    elif peri%n == 0:
        return G
    elif n < len(a):
        for i in range(n):
            temp += -a[i]*x[n-(i+1)]
        return temp
    else:
        for i in range(len(a)):
            temp += -a[i]*x[n-(i+1)]
        return temp

def ar(x,a,n,G):
    temp = 0
    if n == 0:
        return G*np.random.normal(0,1)
    elif n < len(a):
        for i in range(n):
            temp += -a[i]*x[n-(i+1)] + G*np.random.normal(0,1)
        return temp
    else:
        for i in range(len(a)):
            temp += -a[i]*x[n-(i+1)] + G*np.random.normal(0,1)
        return temp












