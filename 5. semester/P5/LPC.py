# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:15:02 2017

@author: Jonas
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
import sys
import os
from scipy import signal
import library as l
import time as time


def LPC(data,fs,s,p,overlap,gender,new):
    data_dec = np.array(data,dtype=np.float64)
    data = data_dec
    
    data = l.PreEmphasis(data_dec,0.95)
    
    #==============================================================================
    # Segmentér og window data
    #==============================================================================
    
    t = time.time()
    Lseg = int(fs*s)
    segments = l.SplitSignal(data,fs,overlap,s)
    segments_dec = l.SplitSignal(data_dec,fs,overlap,s)
    Nseg = int(np.shape(segments)[0])
    for i in range(Nseg):
        segments[i] = segments[i]
    
    #for i in range(Nseg):
    #    segments[i] = segments[i]*np.hamming(Lseg)
    
    #Lseg = np.zeros(Nseg)
    #for i in range(Nseg):
    #    Lseg[i] = int(len(segments[i]))
    segmentingTime = time.time() - t
    print('Segmenting time: %f' %segmentingTime)
                              
    #==============================================================================
    # Ustemt, stemt eller stilhed
    #==============================================================================
    
    t = time.time()
    decision = l.decision_1(segments_dec)
    decisionTime = time.time() - t
    print('Decision time: %f' %decisionTime)
                            
    #==============================================================================
    # Estimér pitch
    #==============================================================================
    
#    decision = np.zeros(Nseg)
#    for i in range(len(decision)):
#        if decision[i] == 0:
#            decision[i] = 2
    
                 
    t = time.time()
    auto = []
    for i in range(Nseg):
        if decision[i] == 2:
            temp = l.estimated_autocorrelation_bias(segments[i])
            
            auto.append(temp)
    
    pitches_r = np.zeros(int(np.shape(auto)[0]))
    
    if gender == 1:
        for i in range(int(len(pitches_r))):
            pitches_r[i] = np.argmax(auto[i][int(fs/185.):int(fs/85.)]) + fs/185.
    else:
        for i in range(len(pitches_r)):
            pitches_r[i] = np.argmax(auto[i][int(fs/255.):int(fs/165.)]) + fs/255.
    
    pitchTime = time.time() - t
    print('Pitch time: %f' %pitchTime)
                         
    #==============================================================================
    # Find filterkoefficienter
    #==============================================================================
    
    t = time.time()
    
    R = []                  # Liste til korrelationsmatricerne
    r = []                  # Liste til korrelationsvektorerne
    a = []                  # Liste til filterkoefficienterne
    u = []                  # Liste til impulstoge
    roots = []              # Liste til rødder af filterpolynomierne
    Ep = np.zeros(Nseg)     # Array til fejlberegningerne
    gain = np.zeros(Nseg)
    index = 0
    for i in range(Nseg):
        temp1, temp2 = l.CorrMatrix(segments[i],p)
        R.append(temp1)
        r.append(temp2)
        a.append(l.LevinsonDurbin(R[i],-r[i],p)[0])
        if decision[i] == 2:
            Ep[i] = pitches_r[index]*l.LevinsonDurbin(R[i],-r[i],p)[2]
            u.append(l.PulseTrain(segments[i],pitches_r[index])*np.sqrt(Ep[i]))
            gain[i] = np.sqrt(Ep[i]*Lseg)
            index += 1
        elif decision[i] == 1:
            Ep[i] = l.LevinsonDurbin(R[i],-r[i],p)[2]
            u.append(np.zeros(Lseg))
            gain[i] = np.sqrt(Ep[i]*Lseg)
        elif decision[i] == 0:
            Ep[i] = l.LevinsonDurbin(R[i],-r[i],p)[2]
            u.append(np.zeros(Lseg))
            gain[i] = np.sqrt(Ep[i])
            
        roots.append(np.roots(np.hstack((1,a[i]))))
    
    LDPTime = time.time() - t
    print('Levinson Durbin and pulse train time: %f' %LDPTime)

    s_tilde = np.zeros(Nseg*Lseg)
        
    s_ar = []
    index = 0

    for i in range(Nseg):
        
        if decision[i] == 2:
            for j in range(int(Lseg)):
                s_tilde[index] = l.AR_V(s_tilde,u[i],a[i],j,index)
                index += 1
        elif decision[i] == 0:
            for j in range(int(Lseg)):
                s_tilde[index] = l.AR_UV(s_tilde,j,a[i],index,gain[i])
                index += 1
        elif decision[i] == 1:
            for j in range(int(Lseg)):
                s_tilde[index] = l.AR_V(s_tilde,u[i],a[i],j,index)
                index += 1
    
    if new == 0:
        for i in range(Nseg):
            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
    else:
        for i in range(Nseg):
            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg-Lseg*overlap])
#    for i in range(Nseg):
#        if decision[i] == 2:
#            s_tilde[i*Lseg:(i+1)*Lseg] = s_tilde[i*Lseg:(i+1)*Lseg]*(np.max(segments[i])/np.max(s_tilde[i*Lseg:(i+1)*Lseg]))
#            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
#        if decision[i] == 0:
#            s_tilde[i*Lseg:(i+1)*Lseg] = s_tilde[i*Lseg:(i+1)*Lseg]*(np.max(segments[i])/np.max(s_tilde[i*Lseg:(i+1)*Lseg]))
#            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
#        else:
#            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
#        elif decision[i] == 1:
#            s_tilde[i*Lseg:(i+1)*Lseg] = np.zeros(Lseg)
#            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])    
    
    #==============================================================================
    # Windowing
    #==============================================================================
    
#    gain = l.RecGain(overlap,Lseg)
#    if overlap != 0:
#        for i in range(Nseg):
#            s_ar[i] = s_ar[i]*gain
    
    #==============================================================================
    # Rekonstruktion
    #==============================================================================
    
#    rec = np.zeros(len(data))
#    index = 0
#    for i in range(Nseg):
#        for j in range(int((1-overlap)*Lseg)):
#            rec[index] = s_tilde[i*Lseg + j]
#            index += 1

    if new == 0:
        rec = l.Reconstruction(s_ar,len(data),fs,overlap,s)
    else:
        rec = np.zeros(0)
        for i in range(Nseg):
            rec = np.hstack((rec,s_ar[i]))
            
        
    #b, a = signal.butter(6,0.015,btype='highpass')
    
    #rec = signal.lfilter(b,a,rec)
    
#    residuals = data - rec
    
    return rec,decision,R,r,a,u,Ep,float(fs)/pitches_r,Nseg,Lseg,segments

def LPC_res(residuals,a,p,overlap,Nseg,Lseg,fs,s,decision):
    
    s_tilde = np.zeros(Nseg*Lseg)
    s_ar = []
    
    index = 0
    for i in range(Nseg):
        if decision[i] != 1:
            for j in range(Lseg):
                s_tilde[index] = l.AR_E(s_tilde,residuals,a[i],index)
                index += 1
            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
        else:
            for j in range(Lseg):
                s_tilde[index] = 0
                index += 1
            s_ar.append(s_tilde[i*Lseg:(i+1)*Lseg])
    rec = l.Reconstruction(s_ar,Nseg*Lseg,fs,overlap,s)
    
    return rec










               


