# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:36:40 2017

@author: Danny
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as siw

#==============================================================================
# Hent data og split det op 
#==============================================================================
fs, wav = siw.read('sounds/a.wav')
wav = wav[:,0]
data = wav
data = np.array(data, dtype = "float64")

N = 160 # samples pr 20ms 
segments = [data[k:k+N] for k in range(0, len(data), N)]

def AMDM(data,k=160):
    N = len(data)
    A = np.zeros(N)
    for t in range(k):
        
        a = 0
        for n in range(N):
            a += np.abs(data[n] - data[n-t])
        A[t] = a
    return np.argmin(A[5:155]) + 5

def auto_corr_meth(k,data,N,delay):
    sum_num = 0
    sum_den = 0
    for n in range(N):
        if n + k < len(data):
            sum_num += data[delay + n]*data[n+delay+k]
            sum_den += data[n+delay+k]**2
        else:
            raise ValueError("Let n+k < N-1")
    return sum_num/np.sqrt(sum_den)

def pitch_esti(data,delay,N):
    R = np.zeros(N)
    for k in range(N):
        R[k] = auto_corr_meth(k,data,N,delay)
    return R

delay = 1000
R_delay = pitch_esti(data,delay,N)
lag = AMDM(data[delay:delay+N],160)


#==============================================================================
# Plots
#==============================================================================


#plt.figure(1)
#plt.plot(data[delay:delay + N])
#plt.show()

#plt.figure(2)
#plt.plot(R_delay)
#plt.show()

print('Pitch ved lag %f' %lag)

#==============================================================================
# Spektralanalyse
#==============================================================================

spek = np.abs(np.fft.fft(data))[:len(data)/2]

























