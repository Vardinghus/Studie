# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:40:07 2017

@author: Frederik Vardinghus
"""

import scipy.io.wavfile as siw
import scipy.signal as scp
import matplotlib.pyplot as plt
import numpy as np
import time
import winsound

#==============================================================================
# Hent data
#==============================================================================
#wav = siw.read('Lydfiler/forsoeg_nopeak/enkelt_tone/forsoeg_enkelt_lys.wav')
#wav = siw.read('Lydfiler/forsoeg_nopeak/melodi/alene/forsoeg_lillepeteredderkop_langsom.wav')
#wav = siw.read('Lydfiler/forsoeg_nopeak/melodi/akkorder/forsoeg_lillepeteredderkop_langsom.wav')
#wav = siw.read('Lydfiler/forsoeg_nopeak/akkorder/forsoeg_akkord_lys.wav')
#wav = siw.read('Lydfiler/forsoeg_nopeak/skala/forsoeg_skala_hurtig.wav')
wav = siw.read('Lydfiler/forsoeg_nopeak/stoej/klap_random_1.wav')
freq1 = wav[0]
data1 = wav[1]
#freq2 = wav2[0]
#data2 = wav2[1]


#if len(data2) < len(data1):
#    data2 = np.lib.pad(data2,(0,len(data1)-len(data2)),'constant',constant_values=0)
#else:
#    data1 = np.lib.pad(data1,(0,len(data2)-len(data1)),'constant',constant_values=0)

freq = freq1
data = data1



plot = 1
time_inter1 = 0
time_inter2 = len(data)
freq_inter1 = 0
freq_inter2 = 20000


save = 0


#data1 = np.zeros(len(data))
#data2 = np.zeros(len(data))
#for i in range(len(data)):
#    data1[i] = data[i][0]
#    data2[i] = data[i][1]
    
t = len(data)/freq
lin = np.linspace(0,t,len(data))

bins = np.linspace(0,freq/2,len(data)/2)

for i in range(len(bins)):
    if bins[i] < freq_inter2:
        freq_inter2_new = i


#plt.plot(lin,data1)

#==============================================================================
# Udtag og plot afsnit af data
#==============================================================================
#N = 2000 # Længde af udsnit
#data = data[0:N]
#lin = lin[0:N]
#plt.plot(lin,data)

#==============================================================================
# Foretag FFT
#==============================================================================
start = time.time()
F = np.abs(np.fft.fft(data)[:len(data)/2.])
end = time.time()
FFT_time = end - start

#==============================================================================
# dB funktion
#==============================================================================
def db(x):
    return 20*np.log10(x)

#==============================================================================
# Udtag og plot afsnit af F
#==============================================================================
#NF = 22050 # Længde af udsnit
#dataF = np.fft.fft(data[0:NF])
#linF = np.linspace(0,NF-1,NF)
    
#plt.plot(linF,dataF)


#==============================================================================
# Frekvensplot og spektrogram genereret med scipy.signal
#==============================================================================
plt.style.use('classic')

fontsize = 15

if plot == 0:
    plt.plot(lin[time_inter1:time_inter2],data[time_inter1:time_inter2])
    plt.legend('Signal')
    plt.ylabel('Amplitude',fontsize=fontsize)
    plt.xlabel('Time [s]',fontsize=fontsize)
elif plot == 1:
    plt.plot(bins[freq_inter1:freq_inter2_new],F[freq_inter1:freq_inter2_new])
    plt.xlabel('Frequency [Hz]',fontsize=fontsize)
    plt.ylabel('Amplitude',fontsize=fontsize)
elif plot == 2:
    plt.specgram(data,Fs=freq)
    plt.xlabel('Time [s]',fontsize=fontsize)
    plt.ylabel('Frequency [Hz]',fontsize=fontsize)



#f,t,Sxx = scp.spectrogram(data,fs=freq)

#plt.pcolormesh(t,f,20*np.log10(Sxx))

#plt.specgram(data,Fs=freq,nperseg=10)
#plt.specgram(data2,Fs=freq)

plt.show()
print 'Mest betydende frekvens:',bins[np.argmax(F)]
print 'Beregningstid for numpy.fft:', FFT_time
print 'Længde af data:',len(data)
print 'Samplingsfrkvens:',freq

if save == 1:
    siw.write('Lydfiler/forsoeg_nopeak/melodi/akkorder/forsoeg_lillepeteredderkop_langsom.wav',freq,data)

#plt.savefig('Lydfiler/figur.png')
#winsound.PlaySound('Lydfiler/forsoeg_nopeak/skala/forsoeg_skala_hurtig.wav', winsound.SND_FILENAME)






