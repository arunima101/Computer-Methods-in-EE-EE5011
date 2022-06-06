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

x=list(np.linspace(0.1,0.9,120))
y=[]
for i in range(len(x)):
    y.append(func(x[i]))

y2a=[0]*len(y)
y3a=s.spline(x,y,100*deri1(0.1),100*deri1(0.9))

xx=list(np.linspace(0.1,0.9,150))
yt=[]
for i in range(len(xx)):
    yt.append(func(xx[i]))

yy=s.splintn(x,y,y3a,xx)


plt.clf()
plt.semilogy(xx,abs(yy-yt),'b')
plt.xlabel("Sample points")
plt.ylabel("Order of error: Logarithmic")
plt.grid()
plt.savefig("Error profile_100xDerivatives")
err=[]
xval=[]
for i in range(50,2000):
    x=list((np.linspace(0.1,0.9,i)))
    y=[]
    for i in range(len(x)):
       y.append(func(x[i]))
    y2a=s.spline(x,y,100*deri1(0.1),100*deri1(0.9))
    xx=list(np.linspace(0.1,0.9,m.floor(2500)))
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
plt.savefig("E6")
'''
c=['b','g','y','c','r','m']
j=0
plt.clf()
for i in range(350,850,100):
    x=list(np.linspace(0.1,0.9,i))
    y=[]
    for i in range(len(x)): 
      y.append(func(x[i]))
    y3a=s.spline(x,y,100*deri1(0.1),100*deri1(0.9))
    xx=list(np.linspace(0.1,0.9,100))
    yt=[]
    for i in range(len(xx)):
      yt.append(func(xx[i]))
    yy=s.splintn(x,y,y3a,xx)
    plt.semilogy(xx,abs(yy-yt),c[j])

    plt.xlabel("Sample values")
    plt.ylabel("Error Order")
    plt.grid()
    plt.savefig("colour2")
    j=j+1






x1=list(np.linspace(0.1,0.9,350))
x2=list(np.linspace(0.1,0.9,450))
x3=list(np.linspace(0.1,0.9,550))
x4=list(np.linspace(0.1,0.9,650))
x5=list(np.linspace(0.1,0.9,750))
x6=list(np.linspace(0.1,0.9,850))
y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
for i in range(len(x1)): 
    y1.append(func(x1[i]))
for i in range(len(x2)): 
    y2.append(func(x2[i]))
for i in range(len(x3)): 
    y3.append(func(x3[i]))
for i in range(len(x4)): 
    y4.append(func(x4[i]))
for i in range(len(x5)): 
    y5.append(func(x5[i]))
for i in range(len(x6)): 
    y6.append(func(x6[i]))

y3a1=s.spline(x1,y1,100*deri1(0.1),100*deri1(0.9))
y3a2=s.spline(x2,y2,100*deri1(0.1),100*deri1(0.9))
y3a3=s.spline(x3,y3,100*deri1(0.1),100*deri1(0.9))
y3a4=s.spline(x4,y4,100*deri1(0.1),100*deri1(0.9))
y3a5=s.spline(x5,y5,100*deri1(0.1),100*deri1(0.9))
y3a6=s.spline(x6,y6,100*deri1(0.1),100*deri1(0.9))

xx=list(np.linspace(0.1,0.9,100))
yt=[]
for i in range(len(xx)):
      yt.append(func(xx[i])) 
yy1=s.splintn(x1,y1,y3a1,xx)
yy2=s.splintn(x2,y2,y3a2,xx)
yy3=s.splintn(x3,y3,y3a3,xx)
yy4=s.splintn(x4,y4,y3a4,xx)
yy5=s.splintn(x5,y5,y3a5,xx)
yy6=s.splintn(x6,y6,y3a6,xx)
plt.semilogy(xx,abs(yy1-yt),label='N=350',color='b')
plt.semilogy(xx,abs(yy2-yt),label='N=450',color='g')
plt.semilogy(xx,abs(yy3-yt),label='N=550',color='y')
plt.semilogy(xx,abs(yy4-yt),label='N=650',color='c')
plt.semilogy(xx,abs(yy5-yt),label='N=750',color='r')
plt.semilogy(xx,abs(yy6-yt),label='N=850',color='m')
plt.legend([abs(yy1-yt),abs(yy2-yt),abs(yy3-yt),abs(yy4-yt),abs(yy5-yt),abs(yy6-yt)],['N=350','N=450','N=550','N=650','N=750','N=850'])
plt.savefig("Colour4")'''

