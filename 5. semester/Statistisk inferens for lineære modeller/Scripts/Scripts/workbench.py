# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:07:16 2017

@author: Frederik Vardinghus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#==============================================================================
# Importér data
#==============================================================================
data = sp.io.loadmat('signal_lp')
data = np.array(data['x'])
signal = np.zeros(len(data))
for i in range(len(data)):
    signal[i] = data[i]

#==============================================================================
# Opgave 1: definér matricer og vectorer
#==============================================================================
M = 4
N = 10

def h(n,x,M):
    h = np.zeros(M)
    for i in range(M):
        h[i] = x[n-i-1]
    return h

row = np.zeros(N)
for i in range(N):
    row[i] = h(i,signal,M)[0]

H = np.transpose(sp.linalg.toeplitz(h(0,signal,M),row))

x = np.zeros((N,1))
for i in range(N):
    x[i][0] = signal[i]

#==============================================================================
# Opgave 4: Simplexalgoritme
#==============================================================================

A = np.hstack((np.vstack((H,-H)),np.vstack((-np.identity(N),-np.identity(N)))))
b = np.vstack((x,-x))
c = np.vstack((np.zeros((M,1)),np.ones((N,1))))



a_k = np.zeros((M+N,1))
a_k[4:] = np.max(b)+1

A_k = np.zeros((0,np.shape(A)[1]))

for y in range(14):
    r = np.round(np.dot(A,a_k) - b,6)
    q = np.min(r)
    for i in range(len(r)):
        if r[i] > 0 and r[i] < q:
            q = r[i]
            l = i
    
    A_star = np.vstack((A_k,A[l]))
    hvec = np.vstack((np.zeros((np.shape(A_k)[0],1)),np.array([[-1]])))
    
    d = np.dot(np.linalg.pinv(A_star),hvec)
    
    I = np.zeros((0,1))
    for i in range(len(b)):
        if np.round(-np.dot(A[i],a_k) - b[i],10) > 0 and np.dot(A[i],d) < 0:
            I = np.vstack((I,i))
        
    I = np.delete(I,0)
    
    alpha = np.zeros((len(I),1))
    for i in range(len(I)):
        alpha[i] = (-np.dot(A[int(I[i])],a_k) - b[int(I[i])])/(-np.dot(A[int(I[i])],d))
    
    i_star = int(I[np.argmin(alpha)])
    
    a_k = a_k - np.min(alpha)*d
    A_k = np.vstack((A_k,A[i_star]))
    
r = np.round(-np.dot(A,a_k) - b,6)
for i in range(len(r)):
    print(r[i])

#for i in range(len(b)):
#    print(np.round(-np.dot(A[i],a_k) - b[i],10))


#l = 0

count = 0
W = range(M+N)
I_full = range(2*N)

#while count <= 100:
#    
#    A_k = A[W]  # Vælg række for aktive restriktioner
#    if np.linalg.det(A_k) == 0:
#        print('A_k er singulær')
#        break
#    
#    a_k = np.linalg.solve(A_k,b[W]) # Find BFP
#    
#    mu = np.linalg.solve(np.transpose(A_k),c)
#    if np.allclose(mu,np.abs(mu)):
#        print(a_k)
#        break
#    
#    for i in W:
#        if mu[i] < 0:
#            l = i
#            break
#    
#    e = np.zeros((len(A_k),1))
#    e[l] = 1
#    
#    d = np.linalg.solve(A_k,e)
#    
#    r = np.dot(A,a_k)-b
#    
#    I = np.zeros(2*N-(M+N))
#    for i in I_full:
#        if W[i] != I_full[i] and np.dot(A[i],d) < 0:
#            I[i] = I_full[i]
#               
#    if len(I) == 0:
#        print('Ubegrænset objektivfunktion')
#        break
#    
#    alpha_vec = np.zeros(len(I))
#    q = 0
#    for i in I:
#        alpha_vec[q] = r[int(i)]/np.dot(-A[int(i)],d)
#        q += 1
#    
#    i_star = int(I[np.argmin(alpha_vec)])
#    alpha = np.min(alpha_vec)
#    
#    a_k = a_k + alpha*d
#    
#    J[l] = i_star
#    
#    
#    count += 1


#for i in range(len(d)):
#    print(np.dot(A[i],d))
#print('r')
#for i in range(len(r)):
#    print(r[i])
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         