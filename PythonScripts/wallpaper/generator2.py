# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 14:54:55 2017

@author: Frederik Vardinghus
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 48
fig_size[1] = 27
plt.rcParams["figure.figsize"] = fig_size

N = 5000
h = 0.5
l = 0.5
p = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
plt.axis('off')

patterns = ['-', '+', 'x', 'o', 'O', '.', '*','/']
x = np.random.rand(N)
y = np.random.rand(N)

colors = np.zeros([N,3])
for i in range(N):
    a = np.random.randint(5)
    if a < 3:
        colors[i][np.random.randint(3)]=1
    elif a == 3:
        colors[i][0]=1
        colors[i][1]=1
    elif a == 4:
        colors[i][1]=1
        colors[i][2]=1
    elif a ==5:
        colors[i][0]=1
        colors[i][2]=1


for i in range(N):
    d = np.random.rand(1)
    ax1.add_patch(
                  patches.Rectangle(
                    (np.random.rand(1), np.random.rand(1)),d*0.07,d*0.07,
                    hatch=patterns[p],
                    facecolor=colors[i],
                    fill='red'))
    if p < len(patterns)-1:
        p += 1
    else:
        p=0


                      
                      
                      
                      
                      
                      