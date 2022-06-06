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

x=list(np.linspace(0.1,0.2,100))
x2=list(np.linspace(0.21,0.4,180))

x3=list(np.linspace(0.41,0.8,285))

x4=list(np.linspace(0.81,0.9,100))

x=x+x2+x3+x4

y=[]
for i in range(len(x)):
    y.append(func(x[i]))
y2a=[0]*len(y)

y2a=s.spline(x,y,deri1(0.1),deri1(0.9))


xx=list(np.linspace(0.1,0.9,1000))
yt=[]
for i in range(len(xx)):
    yt.append(func(xx[i]))
yy=s.splintn(x,y,y2a,xx)
err=max(abs(yy-yt))
plt.clf()
plt.grid()
plt.semilogy(xx,abs(yy-yt),"b")
plt.xlabel("Sample points")
plt.ylabel("Order of error")
plt.savefig("aru trial")

'''
y3a=s.spline(x,y,deri1(0.1),deri1(0.9))
xx1=list(np.linspace(0.5,0.7,10))
yy1=s.splintn(x,y,y3a,xx1)
yout=[]
for i in range(len(xx1)):
    yout.append(func(xx1[i]))
err1=abs(yout-yy1)

plt.semilogy(xx1,err1,"r")
plt.savefig("Trial101")
'''
'''
x=list(np.linspace(0.1,0.2,5))
x2=list(np.linspace(0.21,0.4,25))

x3=list(np.linspace(0.41,0.79,60))

x4=list(np.linspace(0.8,0.9,5))
'''
'''
x=list(np.linspace(0.1,0.2,10))
x2=list(np.linspace(0.21,0.4,25))

x3=list(np.linspace(0.41,0.6,40))
x4=list(np.linspace(0.61,0.79,40))

x5=list(np.linspace(0.8,0.9,10))
x=x+x2+x3+x4+x5

x=list(np.linspace(0.1,0.2,100))
x2=list(np.linspace(0.21,0.4,155))

x3=list(np.linspace(0.41,0.6,125))

x4=list(np.linspace(0.61,0.8,185))
x5=list(np.linspace(0.81,0.9,100))
x=x+x2+x3+x4+x5
'''