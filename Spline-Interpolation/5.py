import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy.special as sp
from os import system
system("f2py -c -m spline spline.f")
import spline as s

def deri1(x):
    y=0.0
    y=-(((x**sp.j0(x))*(-100*x**3 + 2*(100*x**3 - 100*x**2 + x-1)*sp.j0(x) - (2*(100*x**3 - 100*x**2 + x-1)*x*m.log10(x)*sp.j1(x)) +x-2))/(2*(-100*x**3 + 100*x**2 -x +1)**1.5))
    return y

def func(x):
    y1=0.0
    y1=x**(1+sp.j0(x))/(np.sqrt(1-x+100*x**2-100*x**3))
    return y1

x=list(np.linspace(0.1,0.9,17))
y=[]
for i in range(len(x)):
    y.append(func(x[i]))

y2a=[0]*len(y)

y1a=[0]*len(y)
for i in range(len(y)):
    y1a[i]=deri1(x[i])

y2a=s.spline(x,y,deri1(0.1),deri1(0.9))   #method 2 : Using derivatives at the end


xx=list(np.linspace(0.1,0.9,1000))

yt=[]
for i in range(len(xx)):
    yt.append(func(xx[i]))

yy=s.splintn(x,y,y2a,xx)
error=max(abs(yt-yy))
plt.clf()
plt.semilogy(xx,abs(yy-yt),'r')
plt.xlabel("Sample points")
plt.ylabel("Order of error: Logarithmic")
plt.grid()
plt.savefig("Error profile_M2_105 point")

err=[]
xval=[]
for i in range(15,800):
    x=list((np.linspace(0.1,0.9,i)))
    y=[]
    for i in range(len(x)):
       y.append(func(x[i]))
    y2a=s.spline(x,y,deri1(0.1),deri1(0.9))
    xx=list(np.linspace(0.1,0.9,1000))
    yy=s.splintn(x,y,y2a,xx)
    xval.append(i)
    yt=[]
    for i in range(len(xx)):
        yt.append(func(xx[i]))
    err.append(max(abs(yt-yy)))
plt.clf()
plt.semilogy(xval,err,"r")
plt.ylabel("Order of error")
plt.grid()
plt.xlabel("No. of points")
plt.savefig("E5")

