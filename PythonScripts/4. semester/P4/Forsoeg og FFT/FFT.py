

import numpy as np
import matplotlib.pyplot as plt
import time


#==============================================================================
# Parametre
#==============================================================================
N = 2**10 # Antal samples og laengde af FFT
f_s = 2**4   # Samplingsfrekvens
td = 1/float(f_s) # Samplingsperiode
t = td*N

tid = np.linspace(0,t,N) # Samplingspunkter i tid
bins = 2*np.pi*np.linspace(0,1/float(2*td),N/float(2)) # Hoejre halvdel af samplingspunkter i frekvens

#==============================================================================
# Signal
#==============================================================================
def f(x):
    return np.sin(x)

y = f(tid)

#==============================================================================
# DFT
#==============================================================================
def dft(x,c):
    X = np.zeros(c,dtype=complex)
    for k in range(len(x)):
        a = 0+0*1j
        for n in range(c):
            a += x[n]*np.exp(-2*np.pi*1j*k*n/float(c))
            X[k] = a
    return X

#==============================================================================
# FFT
#==============================================================================
def fft(x):
    N_new = len(x)
    if N_new == 2:
        return dft(x) # Returnerer DFT naar data ikke kan deles mere op
    else:
        X_even = fft(x[::2]) # Deler rekursivt input op - lige dele
        X_odd = fft(x[1::2]) # Deler rekursivt input op - ulige dele
        factor = np.exp(-2j * np.pi * np.arange(N_new) / N_new) # Twiddlefaktor
        return np.concatenate([X_even + factor[:N_new / 2] * X_odd,
                               X_even + factor[N_new / 2:] * X_odd])

def ifft(x):
    return np.conj(fft(np.conj(x)))/float(len(y))
    
    
#Y = fft(y)

iters = 22
avg = 10

dft_time = np.zeros(iters)
fft_time = np.zeros(iters)
numpy_time = np.zeros(iters)

samples = np.zeros(iters)

#for i in range(1,iters):
#    print i
#    x = np.linspace(0,1,2**(i))
#    samples[i] = 2**i
#           
#    for k in range(avg):
#        start = time.time()
#        X1 = fft(x)
#        fft_time[i] += time.time() - start
#        start = time.time()
#        X2 = dft(x,2**i)
#        dft_time[i] += time.time()-start
#        start = time.time()
#        X3 = np.fft.fft(x)
#        numpy_time[i] += time.time() - start
#
#    dft_time[i] = dft_time[i]/avg
#    fft_time[i] = fft_time[i]/avg
#    numpy_time[i] = numpy_time[i]/avg

#ratio = fft_time/samples

#plt.plot(samples,dft_time)
#plt.plot(ratio)
#plt.plot(samples,fft_time)
#plt.plot(samples,numpy_time)



x = np.random.rand(126)
g = np.random.rand(126)

conv_t = 0
t1 = 0
t2 = 0
t3 = 0


for i in range(10000):
    start = time.time()
    conv = np.convolve(x,g)
    conv_t += time.time() - start

    start = time.time()
    X = np.fft.fft(np.hstack([x,np.zeros(120)]))
    G = np.fft.fft(np.hstack([g,np.zeros(120)]))
    Y = X*G
    y1 = np.fft.ifft(Y)
    t1 += time.time() - start
                  
    start = time.time()
    X = np.fft.fft(np.hstack([x,np.zeros(125)]))
    G = np.fft.fft(np.hstack([g,np.zeros(125)]))
    Y = X*G
    y2 = np.fft.ifft(Y)
    t2 += time.time() - start
                   
    start = time.time()
    X = np.fft.fft(np.hstack([x,np.zeros(130)]))
    G = np.fft.fft(np.hstack([g,np.zeros(130)]))
    Y = X*G
    y3 = np.fft.ifft(Y)
    t3 += time.time() - start
    

print conv_t
print t1
print t2
print t3








    
    
    
    
    
    

    
    
    