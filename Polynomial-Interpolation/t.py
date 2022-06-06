import matplotlib.pyplot as plt
import math as m
from matplotlib import colors
import numpy as np

import polint as p

x=[0.0,0.0,0.0,0.0,0.0]
y=[0.0,0.0,0.0,0.0,0.0]

for i in range(5):
    if (i!=0):
        x[i]=x[i-1]+0.25
    y[i]=m.sin(x[i]+(x[i])**2)

yy=[p.polint(x,y,w)[0] for w in x]
