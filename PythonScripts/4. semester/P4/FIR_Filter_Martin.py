# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:50:53 2017

@author: Martin Kamp Dalgaard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

M = 92
l = M+1
Fs = 10

N = 2**15
td = 1/float(Fs)
t = td*N

n = np.linspace(0,M,M+1)
x = np.linspace(-np.pi,np.pi,len(n))
xf = np.linspace(0,10*np.pi,len(n))
samples = 2*np.pi*np.linspace(0,t,N)
bins = np.linspace(0,Fs/2,N/2)

delta = (np.pi/15)

f1 = (np.pi/2.) - delta
f2 = (np.pi/2.) + delta

def h(n,M,f1,f2):
    hd = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] == M/2:
            hd[i] = 1 - (f2 - f1)/np.pi
        else:
            hd[i] = (np.sin(f1*(n[i] - M/2.)) / (np.pi*(n[i] - M/2.))) \
            - (np.sin(f2*(n[i] - M/2.)) / (np.pi*(n[i] - M/2.)))
    return hd

def ha(n,M,a): # Hanning window if a = 0.5. Hamming window if a = 0.54.
    w = np.zeros(len(n))
    for i in range(len(n)):
        if abs(x[i]) <= M/2.:
            w[i] = a - (1 - a)*np.cos((2*np.pi*n[i])/M)
        else:
            w[i] = 0
    return w

def blackman(n,M):
    w = np.zeros(len(n))
    for i in range(len(n)):
        w[i] = 0.42 - 0.5*np.cos((2*np.pi*n[i])/M) + 0.8*np.cos((4*np.pi*n[i])/M)
    return w    

w = ha(n,M,0.54)
hd = h(n,M,f1,f2)

def fft(x):
    return np.fft.fft(x)

#hn = np.pad(hd * w,(0,N-M),'constant',constant_values=0)
hn = hd * w

H = np.abs(fft(hn))


a = 2*np.pi

def sig(x):
    return np.sin(np.pi/3.*x) + np.sin(np.pi/2.*x+2*np.pi/3.) + np.sin(3*np.pi/4.*x+4*np.pi/3.)

s = sig(samples)
ideal = sig(samples) - np.sin(np.pi/2.*samples+2*np.pi/3.)

#s_f = np.convolve(s,hn)
#S_F = fft(s)*fft(hn[:N])
#s_f2 = np.fft.ifft(S_F)

tid_inter = 300

#plt.plot(np.abs(np.fft.fft(hd)))

#plt.plot(samples[M/2.:tid_inter],s[0:tid_inter-M/2])
#plt.plot(samples[M/2:tid_inter],ideal[0:tid_inter-M/2])
#plt.plot(samples[M/2.:tid_inter],s_f[M/2.:tid_inter])

#plt.plot(samples[:tid_inter],s_f2[:tid_inter])

#plt.plot(bins,np.abs(fft(s)[0:N/2]/np.max(np.abs(fft(s)))))
#plt.plot(bins,np.abs(fft(hn)[0:N/2]))

plt.figure(2)
plt.plot(x, H)
plt.axis([0,np.pi,0,2])
plt.title('Our bandstop filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude')
plt.axvline(f1*(2*np.pi), color='yellow') # lower cutoff frequency
plt.axvline(f2*(2*np.pi), color='yellow') # upper cutoff frequency
plt.axvline(np.pi/2, color='red') # frequency to be eliminated
plt.axvline(np.pi/3, color='green') # frequency to keep
plt.axvline(3*np.pi/4, color='green') # frequency to keep

#==============================================================================
# Scipy 
#==============================================================================

#omega1_scp = (5*np.pi/12)
#omega2_scp = (5*np.pi/8)
#
#N = [omega1_scp,omega2_scp]
#plt.figure(3)
#b, a = signal.butter(15, N, 'bandstop', analog=True)
#w, h = signal.freqs(b, a)
#plt.plot(w, abs(h), "b-", label = "Bandstop filter")
#plt.title('Scipys bandstop filter frequency response')
#plt.xlabel('Frequency [radians / second]')
#plt.ylabel('Amplitude')
#plt.legend(loc = "lower left")
#plt.margins(0, 0.1)
#plt.axis([0,np.pi,0,2])
#plt.grid(which='both', axis='both')
#plt.axvline(omega1_scp, color='yellow') # lower cutoff frequency
#plt.axvline(omega2_scp, color='yellow') # upper cutoff frequency
#plt.axvline(np.pi/2, color='red') # frequency to be eliminated
#plt.axvline(np.pi/3, color='green') # frequency to keep
#plt.axvline(3*np.pi/4, color='green') # frequency to keep
#plt.show()