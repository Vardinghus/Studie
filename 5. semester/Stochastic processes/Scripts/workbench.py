# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:46:38 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

km = np.linspace(1,21,21)
km = np.hstack((km,0.9))
pace = np.array([370,327,326,332,295,315,307,321,324,307,307,293,304,302,296,309,303,293,309,289,274,276])
elev = np.array([11,-22,13,1,-3,-6,-6,18,-14,-9,17,-3,-20,20,-4,-6,13,-10,6,12,-25,20])

N = len(km)

avgpace = np.zeros(N-1)
avgelev = np.zeros(N-1)
for i in range(N-1):
    avgelev[i] = (elev[i] + elev[i+1])/2
           
avgelev = avgelev/np.max(avgelev)
pace = pace/float(np.max(pace))

corr = np.zeros(N)










