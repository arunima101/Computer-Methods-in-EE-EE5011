import matplotlib.pyplot as plt
import math as m
from matplotlib import colors
import numpy as np

import tableau as p1

x=[0.0,0.0,0.0,0.0,0.0]
y=yp=[0.0,0.0,0.0,0.0,0.0]

for i in range(5):
    if (i!=0):
        x[i]=x[i-1]+0.25
    y[i]=m.sin(x[i]+(x[i])**2)

val=0.89
yp=p1.polint1(x,y,val)[0]
c=np.array(p1.polint1(x,y,val)[2])
d=np.array(p1.polint1(x,y,val)[3])
plt.clf()
plt.grid()
plt.plot(range(0,5),c,'bo',d,'r+')
plt.savefig("Cs and D's")

