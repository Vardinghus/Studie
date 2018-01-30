# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 12:34:40 2017

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
from scipy import signal

#==============================================================================
# Importér data
#==============================================================================

fs, data = wavfile.read('sounds/fr_y.wav',mmap=True)
data = data

#==============================================================================
# Pre-emphasis
#==============================================================================

pre = 0.9
for i in range(len(data)-1):
    data[i+1] = data[i+1] - pre*data[i]

#==============================================================================
# Definér parametre
#==============================================================================

s = 20e-3                           # Længde af segmenter i sekunder
Lseg = int(160)                     # Længde af segmenter i samples
Nseg = int(np.floor(len(data)/Lseg))# Antal segmenter
p = 12                              # Modelorden
bins = np.linspace(0,np.pi,Lseg/2)

#==============================================================================
# Segmentér data
#==============================================================================

segments = np.zeros((Nseg,Lseg))    # Matrix med segmenter
index = 0
for i in range(int(Nseg)):
    for j in range(int(Lseg)):
        segments[i][j] = data[index]
        index += 1

#==============================================================================
# Ustemt, stemt eller stilhed
#==============================================================================



#==============================================================================
# Estimér pitch
#==============================================================================

def AMDM(data):
    """
    Average magnitude difference method
    
    Returnerer lag som giver mindste absolutte
    forskel mellem forskudte samples
    """
    
    N = len(data)
    A = np.zeros(N)
    for t in range(N):
        a = 0
        for n in range(N):
            a += np.abs(data[n] - data[n-t])
        A[t] = a
    return np.argmin(A[int(np.floor(N*0.04)):N-int(np.floor((N*0.04)))]) + N*0.03

def autocorr(x,k):
    N = len(x)
    if np.abs(k) > N - 1:
        return 0
    elif np.abs(k) <= N - 1:
        temp_num = float(0)
        temp_den = float(0)
        for i in range(N-np.abs(k)):
            temp_num += x[i]*x[i+k]
            temp_den += x[i+k]**2
        return temp_num/temp_den

def pitch_esti(data):
    """
    Estimerer pitch i et signal ved brug af
    autokorrelationssekvensen.
    
    Returnerer argument for højeste autokorrelation
    som ikke er r[0].
    """
    
    N = len(data)
    R = np.zeros(N)
    for k in range(N):
        R[k] = autocorr(data,k)
    index = 0
    while R[index] > 0:
        index += 1
    return np.argmax(R[index:len(R)]) + index

pitches_AMDM = np.zeros(Nseg,dtype='int16')
pitches_r = np.zeros(Nseg)
for i in range(Nseg):
    pitches_AMDM[i] = int(AMDM(segments[i]))
    pitches_r[i] = int(pitch_esti(segments[i]))


#==============================================================================
# LPC
#==============================================================================

def levdurp(R,r,n=p):
    r = -r
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
    return a[1:n+1],k[1:n+1],E

def r_bias(x,k):
    N = len(x)
    if k > N - 1:
        return 0
    elif np.abs(k) <= N - 1:
        temp = float(0)
        for i in range(N-k):
            temp += x[i]*x[i+k]
        return temp/N

def corrMatrix(data,p):
    r = np.zeros(p)
    temp = np.zeros(p)
    for k in range(p):
        temp[k] = r_bias(data,k)
        r[k] = r_bias(data,k+1)
    return sp.linalg.toeplitz(temp),-r

def arp(s,u,a,n,index):
    temp = 0
    if index < len(a):
        for i in range(index):
            temp += -a[i]*s[index-(i+1)]
        return temp + u[n]
    else:
        for i in range(len(a)):
            temp += -a[i]*s[index-(i+1)]
        return temp + u[n]

def ar(n,data,a,p=p):
    temp = 0
    if n < p:
        for i in range(n):
            temp += a[i]*data[n-(i+1)]
        return -temp + np.random.normal(0,1)
    else:
        for i in range(p):
            temp += a[i]*data[n-(i+1)]
        return -temp + np.random.normal(0,1)

def pulsetrain(pitch,j):
    u = np.zeros(Lseg)
    for i in range(Lseg):
        if (i-np.argmax(segments[j][:pitch]))%pitch == 0:
            u[i] = 1
    return u

R = []
r = []
a = []
u = []
roots = []
Ep = np.zeros(Nseg)
for i in range(Nseg):
    temp1, temp2 = corrMatrix(segments[i],p)
    R.append(temp1)
    r.append(temp2)
    a.append(levdurp(R[i],r[i])[0])
    u.append(pulsetrain(pitches_AMDM[i],i))
    Ep[i] = levdurp(R[i],r[i])[2]

gain = np.sqrt(Ep*Nseg)
s_tilde = np.zeros(Nseg*Lseg)
index = 0
for i in range(Nseg):
    for j in range(Lseg):
        s_tilde[index] = arp(s_tilde,u[i],a[i],j,index)
        index += 1

#for i in range(Nseg):
#    for j in range(Lseg):
#        s_tilde[index] = ar(index,s_tilde,a[i])
#        index += 1

for i in range(Nseg):
    s_tilde[i*Lseg:(i+1)*Lseg] = s_tilde[i*Lseg:(i+1)*Lseg]*(np.max(segments[i])/np.max(s_tilde[i*Lseg:(i+1)*Lseg]))

n = 9

#plt.figure(1)
#plt.plot(data[n*Lseg:(n+1)*Lseg])
#plt.plot(s_tilde[n*Lseg:(n+1)*Lseg])

#plt.figure(1)
#plt.plot(data)
#plt.plot(s_tilde,alpha=0.8)

#==============================================================================
# Formanter
#==============================================================================

seg = 1

spec = np.abs(np.fft.fft(np.hstack((segments[seg],np.zeros(1024-160)))))[:512]
bins,formant = signal.freqz(b = 1, a = np.hstack((1,a[seg])))
formant = np.abs(formant)
spec = spec/np.max(spec)
formant = formant/np.max(formant)

plt.plot(bins,spec)
plt.plot(bins,formant)

#==============================================================================
# Spektrum
#==============================================================================

#spec_true = np.abs(np.fft.fft(data))[:len(s_tilde)/2]
#spec_tilde = np.abs(np.fft.fft(s_tilde))[:len(s_tilde)/2]
#
#plt.figure(3)
#plt.plot(spec_true)
#plt.figure(4)
#plt.plot(spec_tilde)


#==============================================================================
# Debugging
#==============================================================================

#def tilde(s,a):
#    temp = np.zeros(len(s))
#    for i in range(len(temp)):
#        for k in range(len(a)):
#            temp[i] += -a[k]*s[i-(k+1)]
#    return temp
#            
#s_debug = np.zeros((1,0))
#for i in range(Nseg):
#    s_debug = np.hstack((s_debug,tilde(segments[i],a[i]).reshape(1,160)))
#
#s_debug = np.transpose(s_debug)
#plt.figure(1)
#plt.plot(data[1550:1600])
#plt.plot(s_debug[1550:1600],alpha = 0.6)

#konst = 1
#temp = Ep[konst]
#for k in range(p):
#    temp += a[konst][k]*r[konst][k]
#
#print(temp)
#print(R[konst][0][0])

#==============================================================================
# Gem signal
#==============================================================================

s_tilde = np.array(s_tilde,dtype=np.int16)

wavfile.write('test.wav', fs, s_tilde)


























