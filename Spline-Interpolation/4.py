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
n=len(x)
arr=np.zeros((n-2,n-2))
h=[]
d=[]
for i in range(n-1):
    h.append(x[i+1]-x[i])
for i in range(n-2):
    d.append(6*((y[i+2]-y[i+1])-(y[i+1]-y[i]))/h[0])

for i in range(1,n-3):
    arr[i,i]=2*(h[i-1]+h[i])
    arr[i,i-1]=h[i-1]
    arr[i,i+1]=h[i]
b1=h[0]+2*h[1]
c1=h[1]-h[0]
d[0]=(d[0]*h[1])/2
an=h[n-2]-h[n-3]
bn=2*h[n-3]-h[n-2]
d[n-3]=(d[n-3]*h[n-3])/2
arr[0,0]=b1
arr[0,1]=c1
arr[n-3,n-4]=an
arr[n-3,n-3]=bn

y2a=np.linalg.solve(arr,d)

y3a=[0]*len(y)
y3a[0]=((h[0]+h[1])/h[1])*y2a[0] - (h[0]/h[1])*y2a[1]
y3a[n-1]=((x[n-1]-x[n-3])/(x[n-2]-x[n-3]))*y2a[-1] - ((x[n-1]-x[n-2])/(x[n-2]-x[n-3]))*y2a[-2]
for i in range(len(y2a)):
    y3a[i+1]=y2a[i]
xx=list(np.linspace(0.1,0.9,1000))
yy=s.splintn(x,y,y3a,xx)
yt=[]
for i in range(len(xx)):
    yt.append(func(xx[i]))

plt.clf()
plt.semilogy(xx,abs(yt-yy),'r')
plt.grid()
plt.xlabel('Sample points')
plt.ylabel('Order of error')
plt.savefig("not a knot")

