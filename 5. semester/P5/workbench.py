


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
import sys
import os
from scipy import signal
import LPC as lpc
import library as l
import time as time

dataname = 'sentences'

fs, data = wavfile.read('sounds/%s.wav'%dataname,mmap=True)

#==============================================================================
# Parametre
#==============================================================================

s = 20e-3   
p = 12      # Modelorden
overlap = 0 # Overlap af segmenter
new = 0
gender = 1  # 1 = mand, alt andet = kvinde

#==============================================================================
# LPC
#==============================================================================

rec = lpc.LPC(data,fs,s,p,overlap,gender,new)

rec = rec[0]

rec_de1 = np.int16(l.DeEmphasis(rec,0.95))
wavfile.write('test1.wav',fs,np.int16(rec_de1))



overlap = 0.5 # Overlap af segmenter
new = 1

rec = lpc.LPC(data,fs,s,p,overlap,gender,new)
rec = rec[0]
rec_de2 = np.int16(l.DeEmphasis(rec,0.95))
wavfile.write('test2.wav',fs,np.int16(rec_de2))









