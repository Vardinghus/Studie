# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:47:02 2017

@author: Frederik Vardinghus
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 64
fig_size[1] = 36
plt.rcParams["figure.figsize"] = fig_size

fig, ax = plt.subplots()
patches = []
N = 200
sides = 3

for i in range(N):
    polygon = Polygon(np.random.rand(sides,2), True)
    patches.append(polygon)

p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

colors = 100*np.random.rand(len(patches))
p.set_array(np.array(colors))

ax.add_collection(p)
plt.axis('off')
plt.show()