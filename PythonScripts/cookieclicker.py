# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:40:07 2017

@author: Frederik Vardinghus
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np




ch = 243
th = 268
cb = 14.384

per = (ch**3)/(th**3)*100

defi = (1-per/100)/(per/100)*cb

tb = np.round(cb + defi,2)


print 'Current amount of cookies baked: {0}'.format(cb)
print 'Target heavenly chip: {0}'.format(tb)
print 'Need {0} more cookies to reach total of {1} cookies'.format(np.round(defi,2),tb)







