# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 12:34:40 2017

@author: Jonas
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
import sys
import os

data_name = 'Raw_word'
local_path = os.getcwd()[:-27]
git_path = 'P5\SemesterProjekt\Software\Lyd_Filer\\'
sys.path.insert(0, local_path + git_path)

fs, data = wavfile.read(local_path + git_path + data_name + ".wav",mmap=True)
p = 12
fs = 16000
ms = 20e-3
N = len(data)
Nsamples = float(fs)*ms
MyLen = int(np.ceil(N/Nsamples))
a = np.zeros((MyLen,p))
k = np.zeros((MyLen,p))
E = np.zeros(MyLen)
R = []
r = []

def SplitSignal(data,fs,ms):
    N = len(data)
    Nsamples = float(fs)*ms
    LenAr = np.ceil(N/Nsamples)
    SplitAr = []
    for i in range(int(LenAr)):
        SplitAr.append(np.array(data[int(i*Nsamples):int((i+1)*Nsamples)],dtype=np.int16))
    return SplitAr

def R_bias(X,k):
    N = len(X)
    a = 0
    if k >= 0 and k <= N-1:
        for i in range(N-k-1):
            a += X[i]*X[i+k]
        return a/(N)
    if k >= -(N-1) and k <= -1:
        return R_bias(X,-k)
    if np.abs(k) >= N:
        return 0
    
def ACFfull(data,p,fs,ms):
    N = len(data)
    Nsamples = float(fs)*ms
    MyLen = int(np.ceil(N/Nsamples))
    myACF = np.zeros((MyLen,p))
    for i in range(len(myACF)):
        for k in range(p):
            myACF[i][k] = R_bias(SplitSignal(data,fs,ms)[i],k)
    return myACF

def ACFmatrix(data,k,p = 13,fs = 16000, ms = 20e-3):
    ACF = ACFfull(data,p,fs,ms)
    r = ACF[k][1:]
    R = sp.linalg.toeplitz(ACF[k][:p-1])
    return R,r

def levdurp(R,r,n):
    MyR = np.zeros(n+1)
    for i in range(len(MyR)):
        if i < n:
            MyR[i] = R[0,i]
        if i == n:
            MyR[i] = r[-1]
    k = np.zeros(n+1)
    a = np.zeros(n+1)
    p = 0
    if p == 0:
        E = MyR[0]
        k[p] = 1
        a[0] =1
        p = p +1
    while p <= n:
        q = 0
        for i in range(0,p):
            q += a[i]*MyR[p-i]
        k[p] = (-q)/E
        a_copy = np.copy(a)
        a[p] = k[p]
        for i in range(1,p):
            a[i] = a_copy[i] + k[p]*a_copy[p-i]
        E = E*(1-k[p]**2)
        p= p+1
    return -a[1:n+1],-k[1:n+1],E

for i in range(MyLen):
    R.append(ACFmatrix(data,i)[0])
    r.append(ACFmatrix(data,i)[1])



for j in range(MyLen):
    a[j] = levdurp(R[j],r[j],p)[0]
    k[j] = levdurp(R[j],r[j],p)[1]
    E[j] = levdurp(R[j],r[j],p)[2]

Gain = np.sqrt(E)
plt.plot(data)


