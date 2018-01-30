# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:15:02 2017

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
import sys
import os
from scipy import signal
import time as time
import library as l

#==============================================================================
# 
#==============================================================================

def autocorr(x,k,N):
    temp_num = float(0)
    temp_den = float(0)
    for i in range(N):
        temp_num += x[i]*x[i+k]
        temp_den += x[i+k]**2
    return temp_num/np.sqrt(temp_den)

def ACS_bias(x,k):
    """
    x = signal
    k = lag
    
    Returnerer biased sample ACS ved lag k.
    """
    N = len(x)
    if np.abs(k) > N - 1:
        return 0
    elif np.abs(k) <= N - 1:
        temp = float(0)
        for i in range(N-np.abs(k)-1):
            temp += x[i]*x[i+k]
        return temp/N

def periodogram(data):
    return np.abs(np.fft.fft(data))**2

def invPeriodogram(data):
    return np.fft.ifft(data)*np.bartlett(2*len(data))[len(data):]

def fejer(x,N):
    return (np.sin(np.pi*x*N)/np.sin(np.pi*(x+10e-6)))**2

def estimated_autocorrelation(x):
    n = len(x)
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/np.arange(n, 0, -1)*np.bartlett(2*n)[n:]
    return result

#==============================================================================
# 
#==============================================================================

N = 2000

lin = np.linspace(0,0.05,N)
x = np.sin(200*2*np.pi*lin)

rt = np.zeros(N)
t = time.time()
for j in range(N):
    rt[j] = ACS_bias(x,j)
acstime = time.time() - t

                   
t = time.time()
X = np.fft.fft(x)
S = X*np.conjugate(X)
rf = np.real(np.fft.ifft(S))
periodogramTime = time.time() - t

t = time.time()
snyd = estimated_autocorrelation(x)
snydtime = time.time() - t            
ratio = acstime/(snydtime+10e-13)
      
snyd = snyd/np.max(snyd)
i = 0
while snyd[i] > 0:
    i += 1          

X = lin[np.argmax(snyd[i:])] + lin[i]


#g = np.linspace(1,0.65,N)
#snyd = g*snyd
                           
#==============================================================================
#
#==============================================================================
                   
#print('Time for temporal autocorrelation: %f' %acstime)
#print('Time for snyd: %f' %snydtime)
#print('Ratio: %f' %ratio)

plt.figure(1)
plt.title('Sine, 200 [Hz]',fontsize=16)
plt.xlabel('Amplitude',fontsize=13)
plt.ylabel('Time [s]',fontsize=13)
plt.plot(lin,x/np.max(x))
plt.figure(2)
plt.title('Biased sample ACS',fontsize=16)
plt.xlabel('Correlation',fontsize=12)
plt.ylabel('Lag [s]',fontsize=12)
plt.plot(lin,snyd,label='Autocorrelation')
plt.plot(X,np.max(snyd[i:]),'o',label='Estimated pitch')
plt.legend()



