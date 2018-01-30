# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:52:26 2017

@author: Frederik Vardinghus
"""

from __future__ import division
import numpy as np
from scipy.io import wavfile
import scipy as sp
from scipy import signal
import library as l

def formant(a,b=1,p=12):
    """ Finder |H(e^(jw)|), frekvensbins og formanterne
    Input a fra Ra=-r
    Output w,|H(e^(jw)|),Formanter"""
    a_ny = np.hstack((1,a))
    w,H = signal.freqz(b,a_ny)
    H = abs(H)
    roots = rootsFilter(a)
    
    half_root_temp = []
    for i in range(len(roots)):
        if np.allclose(np.imag(roots[i]),0) != True:
            half_root_temp.append(roots[i])
    half_root = half_root_temp[::2]
    temp = 0
    idx = []
    for i in range(len(roots)):
        if np.allclose(np.imag(roots[i]),0):
            temp += 1
            idx.append(i)
    for i in range(temp):
        if np.real(roots[idx[i]]) > 0:
            half_root = np.append(0,half_root)
        if np.real(roots[idx[i]]) < 0:
            half_root = np.append(half_root,-1)
    RootArg = np.zeros(len(half_root))
    
    for i in range(len(RootArg)):
        RootArg[i] = np.angle(half_root[i])
        if RootArg[i] == np.pi:
            RootArg[i] = np.pi-0.01
        RootArg[i] = int((RootArg[i]/np.pi)*(2**9))
    formants = np.zeros((int(len(half_root)),2))
    for i in range(len(formants)):
        formants[i][0] = w[int(RootArg[i])]
        formants[i][1] = H[int(RootArg[i])]/max(H)
    return w,H,formants

def spectrum(signal):
    """Forlænger signalet til længde 2**9, så
        frekvens opløsningen bliver bedre"""
    N = len(signal)
    N_rest = 2**10-N
    signal = np.append(signal,np.zeros(N_rest))
    return np.abs(np.fft.fft(signal)[0:2**9])

def rootsFilter(a):
    """Finder poler i vores filter
        Input a fra Ra=-r
        Output rødderne"""
    temp = np.append(1,a)
    return np.roots(temp)

def circ():
    N = 100
    t = np.linspace(0,np.pi*2,N)
    circle_x = np.zeros(N)
    circle_y = np.zeros(N)
    for i in range(len(t)):
        circle_x[i],circle_y[i] = np.sin(t[i]),np.cos(t[i])
    return circle_x,circle_y


def FreqBins(N, fs):
    """
    N = antal samples i signal
    fs = samplingsfrekvens
    
    Returnerer liste med frekvensbins fra 0 til samplingsfrekvensen.
    """
    f_bin = [k*fs/float(N) for k in range(N)]
    return f_bin

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

def CorrMatrix(data,p):
    """
    data = signal
    p = modelorden
    
    Returnerer to arrays:
        - En Toeplitzmatrix indeholdende biased sample ACS
        for lag fra 0 til og med p-1 (autokorrelationsmatricen).
        - En vektor indeholdende biased sample ACS for lag
        fra 1 til og med p.
    """ 
    r = np.zeros(p)
    temp = np.zeros(p)
    for k in range(p):
        temp[k] = ACS_bias(data,k)
        r[k] = ACS_bias(data,k+1)
    return sp.linalg.toeplitz(temp),r

def CorrFull(data,Nseg,p):
    """
    data = signal
    Nseg = antal segmenter som signalet splittes op i
    p = modelorden
    
    Returnerer to arrays:
        - En liste med samtlige korrelationsmatricer fra
        CorrMatrix tilhørende segmenterne af signalet.
        - En liste med samtlige vektorer indeholdende
        autokorrelationskoefficienter fra 1 til og med
        p tilhørende segmenter af signalet.
    """
    R_full = []
    r_full = []
    for i in range(Nseg):
        r_full.append(CorrMatrix(SplitSignal(data)[i],p)[1])
        R_full.append(CorrMatrix(SplitSignal(data)[i],p)[0])
    return R_full, r_full

def LevinsonDurbin(R,r,p):
    """
    R = Toeplitz autokorrelationsmatrix (0 til og med p-1)
    r = autokorrelationskoefficienter (1 til og med p)
    p = modelorden
    
    Returnerer tre ting:
        - Løsningen a til ligningssystemet R*a = -r.
        - Hjælpekoefficienterne k.
        - Den minimerede kvadrede fejl ved estimering af a.
    """
    r = -r
    MyR = np.zeros(p+1)
    for i in range(len(MyR)):
        if i < p:
            MyR[i] = R[0,i]
        if i == p:
            MyR[i] = r[-1]
    k = np.zeros(p+1)
    a = np.zeros(p+1)
    n = 0
    if n == 0:
        E = MyR[0]+10e-10
        k[p] = 1
        a[0] =1
        n = n +1
    while n <= p:
        q = 0
        for i in range(0,p):
            q += a[i]*MyR[n-i]
        k[n] = (-q)/E
        a_copy = np.copy(a)
        a[n] = k[n]
        for i in range(1,n):
            a[i] = a_copy[i] + k[n]*a_copy[n-i]
        E = E*(1-k[n]**2)
        n = n+1
    return a[1:n+1],k[1:n+1],E

def PreEmphasis(data,a):
    """
    data = signal
    a = filterkoefficient -- typisk i intervallet [0.9;0.98]
    
    Returnerer højpasfiltreret signal.
    """
    temp = np.zeros(len(data))
    temp[0] = data[0]
    for i in range(1,len(data)):
        temp[i] = data[i] - a*data[i-1]
    return temp

def DeEmphasis(data,a):
    """
    data = signal
    a = filterkoefficient -- typisk i intervallet [0.9;0.98]
    
    Returnerer højpasfiltreret signal.
    """
    temp = np.zeros(len(data))
    temp[0] = 0
    for i in range(1,len(data)):
        temp[i] = data[i] + a*temp[i-1]
    return temp

def PulseTrain(data,pitch):
    """
    data = signal som der skal laves impulstog til
    pitch = pitch data
    
    Returnerer et impulstog med angivet pitch.
    """
    u = np.zeros(len(data))
    for i in range(len(data)):
        if (i-np.argmax(data[:int(pitch)-1]))%int(pitch) == 0:
            u[i] = 1
    return u

def AR_UV(data,n,a,index,gain):
    """
    data = signal som der modelleres som autoregressiv process
    n = indeks som ønskes beregnet (prædikteret)
    a = koefficienter for AR-process (filterkoefficienter)
    
    Returnerer næste værdi for den autoregressive process
    prædikteret fra de foregående adderet hvid støj.
    Denne bruges til ustemt tale.
    """
    temp = 0
    if index < len(a):
        for i in range(len(a)):
            temp += a[i]*data[n-(i+1)]
        return -temp + np.random.normal(0,1)*gain
    else:
        for i in range(len(a)):
            temp += a[i]*data[n-(i+1)]
        return -temp + np.random.normal(0,1)*gain

def AR_V(data,u,a,n,index):
    """
    data = signal
    u = impulstog med pitch for signal
    a = koefficienter for AR-process (filterkoefficienter)
    n = indeks som ønskes beregnet (prædikteret)
    
    Returnerer næste værdi for den autoregressive process
    prædikteret fra de foregående evt. adderet en impuls.
    Denne bruges til stemt tale.
    """
    temp = 0
    if index < len(a):
        for i in range(index):
            temp += -a[i]*data[index-(i+1)]
        return temp + u[n]
    else:
        for i in range(len(a)):
            temp += -a[i]*data[index-(i+1)]
        return temp + u[n]
    
def AMDM(data,fs,gender = 'M'):
    if gender == 'M':
        lb = (fs/200)
    if gender == 'F':
        lb = (fs/300)
    if (gender != 'M') and (gender != 'F'):
        raise NameError('That gender is unknown to me')
    
    """
    Average magnitude difference method
    
    Returnerer lag som giver mindste absolutte
    forskel mellem forskudte samples.
    """

    N = len(data)
    A = np.zeros(N)
    for t in range(int(N)):
        a = 0
        for n in range(N):
            a += np.abs(data[n] - data[n-t])
        A[t] = a
    return A

def autocorr(x,k,N):
    temp_num = float(0)
    temp_den = float(0)
    for i in range(N):
        temp_num += x[i]*x[i+k]
        temp_den += x[i+k]**2
    return temp_num/np.sqrt(temp_den)

def pitch_r(data):
    N = len(data)
    R = np.zeros(N)
    g = np.linspace(0,0.65,N/2)
    
    for k in range(int(N/2)):
        R[k] = g[k]*autocorr(data,k,int(N/2))
    index = 0
    while R[index] > 0:
        index += 1
    return np.argmax(R[index:]) + index

def estimated_autocorrelation_bias(x):
    n = len(x)
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/np.arange(n, 0, -1)*np.bartlett(n*2)[n:]
    return result  
               
def SplitSignal(data,fs,overlap,s):
    """
    data = signal
    fs = samplingsfrekvens
    s = segmentlængde i sekunder
    
    Returnerer liste indeholdende segmenter af signalet
    svarende til længden angivet i sekunder.
    """
    Lseg = int(fs*s)
    SplitAr = []
    inc = (1-overlap)*Lseg
    i = 0
    while i*inc + Lseg <= len(data):
        SplitAr.append(data[int(i*inc):int(i*inc+Lseg)])
        i += 1
    return np.array(SplitAr).reshape(np.shape(SplitAr)[0],np.shape(SplitAr)[1])

def Reconstruction(Signal,length,fs,overlap,s):
    """
    Signal er en matrix beståenden af de overlappende segmenter
    fs  er samplingsfrekvens
    overlap er hvor meget overlap der bliver kørt med typisk 0.5 med et hamming vindue
    
    Retunere et array med det rekonstruerede signal
    """
    
    Lseg = int(s*fs)# Længden af segmenterne
    R = int((1-overlap)*Lseg) # Det antal samples vinduet bliver forskudt med
    Nr = np.shape(Signal)[0] # Antallet af vinduer
    w = np.hamming(Lseg)
    x = np.zeros(length)
    xr = np.array(Signal).reshape(np.shape(Signal)[0],np.shape(Signal)[1])
    for r in range(Nr):
        for n in range(R):
            n = int(n + r*R)
            indx = int(n-r*R)
            x[n] = xr[r,indx]
    return x              
         
def RecGain(overlap,wlen):
    if wlen%2 != 0:
        R = int((1-overlap)*(wlen-1))
    else:
        R = int((1-overlap)*wlen)
    w = np.zeros(25*wlen)
    r = 0
    while wlen + r*R <= len(w):
        for n in range(wlen):
            w[int(n+r*R)] += np.hamming(wlen)[n]
        r += 1
    return np.max(w)
      
#==============================================================================
#==============================================================================
# Decision algorithm     
#==============================================================================
#==============================================================================

#==============================================================================
# Functions for determination of classification parameters for each segment
#==============================================================================
def zerocros(data): 
    """
    Zerocross count for one segment 
    Input: one segment as array. 
    Output: one integer value
    """
    c=0
    for i in range(len(data)-1):
        if data[i]*data[i+1]< 0:
            c+=1    
    return c #normalisation is removed here

def autocor_coef(data):
    """
    Aotocorrelation coefficient for one segment 
    Input: one segment as array. 
    Output: one integer value
    """
    N=len(data)
    s1=0
    s2=0
    s3=0
    for n in range(1,N):
        s1 += data[n]*data[n-1]
        s2 += data[n]**2
    for n in range(N-1):
        s3 += data[n]**2
    
    return (s1)/(np.sqrt(s2*s3)+10e-6)
    
def energy(data):
    """
    Average log energy for one segment 
    Input: one segment as array. 
    Output: one integer value
    """
    e=0
    for i in range(len(data)):
        e += np.abs(data[i])**2
        avg = 10*np.log10(10e-6+(1/len(data)*(e)))
    return avg

def pre_error(data,p=12):
    """
    Prediction error for one segment. Determined by Durbin Levison algorithm.    
    Input: one segment as array, filter order. 
    Output: one integer value
    """

    R,r = l.CorrMatrix(data,p)
    Ep = l.LevinsonDurbin(R,-r,p)[2]
    E = energy(data)

    LogE = E-(10*np.log10(10E-6+Ep))
    return LogE
    
def pre_coef(data,p):
    r=[]
    R=[]
    
    R,r = CorrMatrix(data,p)
    
    a = LevinsonDurbin(R,-r,p)[0]
    
    return a[0] 
    
def check_nan(x,x2,name):
    
    for i in range(len(x)):
            if np.isnan(x)[i]==True or np.isnan(x2)[i]==True  :
                print 'ERROR: %s contain nan '%(name)
                break 
    return

#==============================================================================
# Decision algortihm 1. "Weighted combinations"
##==============================================================================
def decision_1(segments):
    """
    "Weighted combination" decision algortihm for classification of UV/S/V.
    Input:      Matrix with each segment as a row 
    Output:     Array of lenght(number of segments). 
                where 0 = unvoiced, 1 = silence, 2 = voiced
    
    Threshold and weights are to be ajusted in the input of function S and V.
    """
    def norm_ZCR(TH1,segments):
        ZCR = np.zeros(len(segments))
        ZCR2 = np.zeros(len(segments))
        for i in range(len(segments)):
            ZCR[i] = zerocros(segments[i])
            if TH1 > ZCR[i]:
                ZCR2[i] = float(TH1 - ZCR[i])/float((TH1 - np.min(ZCR))+10e-6)
            else:
                ZCR2[i] = float(TH1 - ZCR[i])/float((np.max(ZCR) - TH1)+10e-6)
        #check for nan
        check_nan(ZCR,ZCR2,'ZCR')
        return ZCR2
    
    def norm_LogE(TH2,segments):
        LogE  = np.zeros(len(segments))
        LogE2 = np.zeros(len(segments))
        for i in range(len(segments)):
            LogE[i] = energy(segments[i])
            if TH2 < LogE[i]:
                LogE2[i] = (float(LogE[i] - TH2))/(float(np.max(LogE) - TH2))
            else: 
                LogE2[i] = (float(LogE[i] - TH2))/(float(TH2 - np.min(LogE)))
        if 0 in LogE:
            print 'fejl i LogE'
        #check for nan
        check_nan(LogE,LogE2,'LogE')
        return LogE2
        # swich sign for silence algorithm
    
    def norm_AC(TH3,segments):
        AC = np.zeros(len(segments))
        AC2 = np.zeros(len(segments))
        for i in range(len(segments)):
            AC[i] = autocor_coef(segments[i])
            if TH3 > AC[i]:
                AC2[i] = (AC[i]- TH3)/(TH3 - np.min(AC))
            else: 
                AC2[i] = (AC[i] - TH3)/(max(AC) - TH3)
        
        #check for nan
        check_nan(AC,AC2,'AC')
        return AC2
    
    def error_norm(TH4,segments):  
        log_error = np.zeros(len(segments))
        log_error2 = np.zeros(len(segments))
        for i in range(len(segments)):
            log_error[i] = pre_error(segments[i])
            
            if TH4 < log_error[i]:
                log_error2[i]=float(log_error[i]- TH4)/float(np.max(log_error)-TH4)
            else:
                log_error2[i]=float(log_error[i] - TH4)/float(TH4 - np.min(log_error))  
        #check for nan
        check_nan(log_error,log_error2,'error')  
        return log_error2
        
    def coef_norm(TH5,segments):
        coef = np.zeros(len(segments))
        coef2 = np.zeros(len(segments))
        for i in range(len(segments)):
            coef[i] = pre_coef(segments[i],12)
            if TH5 > coef[i]:
                coef2[i] = (TH5-coef[i])/(TH5 - np.min(coef))
            else: 
                coef2[i] = (TH5-coef[i])/(max(coef) - TH5) 
        #check for nan
        check_nan(coef,coef2,'error')  
        return coef2
    
    #def S(segments,THs1=60,THs2=30,THs3=0.85,THs4=45,THs5=-0.47,k1=0.2,k2=0.5,k3=0.2,k4=0.1,k5=0.0): # for letter signal
    #def S(segments,THs1=33,THs2=43,THs3=0.8,THs4=8.8,THs5=-0.55,k1=0.2,k2=0.5,k3=0.2,k4=0.1,k5=0.0): # for 10 [ms]
    def S(segments,THs1=56,THs2=44,THs3=0.8,THs4=9.6,THs5=-0.5,k1=0.2,k2=0.5,k3=0.2,k4=0.1,k5=0.0):  # for 20 [ms]
        #set thersholds and weights here
        ZCR_n = norm_ZCR(THs1,segments)
        E_n = -(norm_LogE(THs2,segments)) 
        AC_n = -(norm_AC(THs3, segments))
        Ep_n = -(error_norm(THs4, segments))
        coef_n = -(coef_norm(THs5, segments))
        
        cl1=np.zeros(len(segments))
        for i in range(len(segments)):
            cl1[i] = k1*ZCR_n[i]+k2*E_n[i]+k3*AC_n[i]+k4*Ep_n[i]+k5*coef_n[i]
        return cl1,THs1,THs2,THs3,THs4,THs5
    
    #def V(segments,TH1=30,TH2=45,TH3=0.5,TH4=45,TH5=-0.86,c1=0.3,c2=0.3,c3=0.3,c4=0.1,c5=0.0):  # for letter signal  
    #def V(segments,TH1=24,TH2=56,TH3=0.6,TH4=10,TH5=-0.88,c1=0.3,c2=0.2,c3=0.3,c4=0.2,c5=0.0): # for 10 [ms]
    def V(segments,TH1=48,TH2=56,TH3=0.6,TH4=11,TH5=-0.9,c1=0.3,c2=0.2,c3=0.3,c4=0.2,c5=0.0):  # for 20 [ms]
        #set thersholds and weights here    
        ZCR_n = norm_ZCR(TH1,segments)
        E_n = norm_LogE(TH2,segments)
        AC_n = norm_AC(TH3, segments)
        Ep_n = error_norm(TH4, segments)
        coef_n = coef_norm(TH5, segments)
        
        cl2=np.zeros(len(segments))
        for i in range(len(segments)):
            cl2[i] = c1*ZCR_n[i]+c2*E_n[i]+c3*AC_n[i]+c4*Ep_n[i]+c5*coef_n[i]
        return cl2,TH1,TH2,TH3,TH4,TH5
    
       
    s,THs1,THs2,THs3,THs4,THs5 = S(segments)
    v,TH1,TH2,TH3,TH4,TH5 = V(segments)
    deci = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] > 0: 
            deci[i]=1
        else:
            if v[i] > 0:
                deci[i] = 2
            else:
                deci[i] = 0
            
    return deci # 0 = unvoiced, 1 = silence, 2 = unvoiced


#==============================================================================
# Decision algorithm 2 "minimum distance"
#==============================================================================
def decision_2(segments,L=4):
    """
"Weighted combination" decision algortihm for classification of UV/S/V.
Input:      Matrix with each segment as rows 
Output:     Array of lenght(number of segments). 
            where 0 = unvoiced, 1 = silence, 2 = voiced
"""
    classes=3   #number of classes
    
    # define W_i and m_i, according to given training set 
    # mi = [ZCR, E_s, rho] 
    # 0 = unvoiced, 1 = silence, 2 = voiced 
    
    # mi = [ZCR, E_s, rho, E_p] 
    
    m0 = np.array([49.914,23.439,0.007,3.661])
    W0 = np.matrix([[12.680**2,0.471,-0.959,-0.019],
                    [0.471,6.985**2,-0.454,0.447],
                    [-0.959,-0.454,0.365**2,0.028],
                    [-0.019,0.447,0.028,1.763**2]])
    
    
    m1 = np.array([25.663,10.781,0.649,4.976]) 
    W1 = np.matrix([[7.534**2,-0.032,-0.842,-0.629],
                    [-0.032,4.715**2,-0.098,0.580],
                    [-0.842,-0.098,0.158**2,0.596],
                    [-0.629,0.580,0.596,1.994**2]])
    
    
    m2 = np.array([12.775,50.608,0.881,18.944])
    W2 = np.matrix([[5.546**2,0.250,-0.882,-0.626],
                    [0.250,5.530**2,-0.200,-0.051],
                    [-0.882,-0.200,0.090**2,0.728],
                    [-0.626,-0.051,0.728,6.151**2]])
    
    # compute x from parameters
    E=np.zeros(len(segments))
    ZCR=np.zeros(len(segments))
    rho=np.zeros(len(segments))
    Ep= np.zeros(len(segments))
    for i in range(len(segments)):
        E[i] = energy(segments[i])
        ZCR[i] = zerocros(segments[i])
        rho[i] = autocor_coef(segments[i])
        Ep[i] = pre_error(segments[i])
        
    x = np.zeros([len(segments),L])
    for j in range(len(segments)):
        x[j] = np.array([ZCR[j],E[j],rho[j],Ep[j]])
    
    # compute deviance
    d = np.zeros((len(segments),classes))
    for i in range(len(segments)):
        d[i][0] = np.dot(np.dot(np.transpose(x[i]-m0),np.linalg.inv(W0)),(x[i]-m0))
        d[i][1] = np.dot(np.dot(np.transpose(x[i]-m1),np.linalg.inv(W1)),(x[i]-m1))
        d[i][2] = np.dot(np.dot(np.transpose(x[i]-m2),np.linalg.inv(W2)),(x[i]-m2))
    
    # make decision
    deci = np.zeros(len(d))    
    for j in range(len(d)):
        deci[j]=np.argmin(np.abs(d[j]))
    
    return deci # 0 = unvoiced, 1 = silence, 2 = unvoiced




