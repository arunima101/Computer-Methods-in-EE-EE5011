import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from os import system
system("f2py -c -m spline_modified spline_modified.f")
import spline_modified as s1
system("f2py -c -m spline spline.f")
import spline as s
from scipy.interpolate import CubicSpline


xa=np.linspace(0,10,11)
ya=np.sin(xa)
y2a=np.zeros(ya.shape)
xx=np.linspace(-5,15,101)
yt=np.sin(xx)
yy=np.zeros(xx.shape)
u=np.zeros(ya.shape)

y2a=s.spline(xa,ya,1e40,1.e40)

print( np.c_[xa,ya])
plt.clf()
plt.figure(1)
plt.plot(xa,ya,'r',xa,y2a,'g')
plt.legend(["ya","y2a"])
plt.grid(True)
plt.savefig("1.jpg")

yyb=s.splintn(xa,ya,y2a,xx)

plt.clf()
plt.figure(2)
plt.plot(xa,ya,'ro',xx,yt,'b+',xx,yyb,'k')
plt.xlim([-5,15])
plt.ylim([-10,10])
plt.grid(True)
plt.savefig('2.jpg')
plt.clf()
plt.semilogy(xx,abs(yt-yyb),'g')
plt.grid(True)
plt.savefig("3.jpg")





y3a=s1.spline(xa,ya,1.0,np.cos(10.0))
yyc=s1.splintn(xa,ya,y3a,xx)
plt.clf()
plt.semilogy(xx,abs(yyc-yt),'b',xx,abs(yt-yyb),'g')
plt.savefig("5.jpg")