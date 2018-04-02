


import numpy as np
import matplotlib.pyplot as plt


def isprime(number):
	if number<=1 or number%2==0:
		return 0
	check=3
	maxneeded=number
	while check<maxneeded+1:
		maxneeded=number/check
		if number%check==0:
			return 0
		check+=2
	return 1

def tabel(number,tabel):
    if number % tabel == 0:
        return 1
    else:
        return 0

size = 200

x = np.zeros(size**2)
y = np.zeros(size**2)
color = []
marker = []
alpha = []

x[0] = size/2
y[0] = size/2

xcount = 1
ycount = 1
index = 0

while index + 1 < size**2:
    
    for j in range(xcount):
        x[index+1] = x[index] + 1
        y[index+1] = y[index]
        index += 1
        
        if index + 1 == size**2:
            break

    if index + 1 == size**2:
        break
    xcount += 1
    
    for j in range(ycount):
        x[index + 1] = x[index]
        y[index + 1] = y[index] + 1
        index += 1
        
        if index + 1 == size**2:
            break
        
    if index + 1 == size**2:
        break
    ycount += 1
    
    for j in range(xcount):
        x[index+1] = x[index] - 1
        y[index+1] = y[index]
        index += 1
        
        if index + 1 == size**2:
            break
        
    if index + 1 == size**2:
        break
    xcount += 1
    
    for j in range(ycount):
        x[index + 1] = x[index]
        y[index + 1] = y[index] - 1
        index += 1
        
        if index + 1 == size**2:
            break
        
    if index + 1 == size**2:
        break

    ycount += 1

index = 0

# Primtal
for i in range(size**2):
    if isprime(i) == 1:
        color.append('r')
    else:
        x = np.delete(x,i-index)
        y = np.delete(y,i-index)
        index += 1

plt.scatter(x,y,c=color,marker='.')














