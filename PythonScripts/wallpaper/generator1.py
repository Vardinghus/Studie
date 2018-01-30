# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:59:52 2017

@author: Frederik Vardinghus
"""

import matplotlib.pyplot as plt
import numpy as np

# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
## Set figure width to 12 and height to 9
#fig_size[0] = 40
#fig_size[1] = 22.5
#plt.rcParams["figure.figsize"] = fig_size

N = 200
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=1)
plt.axis('off')
plt.show()
