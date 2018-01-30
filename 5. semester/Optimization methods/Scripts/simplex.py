# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 01:05:44 2017

@author: Frederik Vardinghus
"""

import numpy as np

c = np.array([[-1],
              [-4]])
A = np.array([[1,0],
             [-1,0],
             [0,1],
             [-1,-1],
             [-1,-2]])
b = np.array([[0],
              [-2],
              [0],
              [-3.5],
              [-6]])

count = 0
J = [0,3]

while count <= 100:
    
    A_k = A[J,]
    if np.linalg.det(A_k) == 0:
        print('A_k er singulær')
        break
    
    a_k = np.linalg.solve(A_k,b[J])
    
    mu = np.linalg.solve(np.transpose(A_k),c)
    if np.allclose(mu,np.abs(mu)):
        print(a_k)
        break
        
    l = np.argmin(mu)
    e = np.zeros((len(A_k),1))
    e[l] = 1
    
    d = np.linalg.solve(A_k,e)
    
    r = np.dot(A,a_k)-b
    
    a = 0
    for i in range(len(r)):
        if r[i] > 0 and np.dot(A[i],d) < 0:
            a += 1
    I = np.zeros(a)
    q = 0
    for i in range(len(r)):
        if r[i] > 0 and np.dot(A[i],d) < 0:
            I[q] = i
            q += 1
    
    if len(I) == 0:
        print('Ubegrænset objektivfunktion')
        break
    
    alpha_vec = np.zeros(len(I))
    q = 0
    for i in I:
        alpha_vec[q] = r[int(i)]/np.dot(-A[int(i)],d)
        q += 1
    
    i_star = int(I[np.argmin(alpha_vec)])
    alpha = np.min(alpha_vec)
    
    a_k = a_k + alpha*d
    
    J[l] = i_star
    
    
    
    count += 1






