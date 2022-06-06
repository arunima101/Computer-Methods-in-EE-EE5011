from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.special as sp
from os import system
system("f2py -c -m spline spline.f")
import spline as s

def func(x,y):
    z=0.0
    z=(np.sin(np.pi*x))*(np.cos(np.pi*y))
    return z

def deri2(x,y):
    z=0.0
    z=-(np.pi**2)*(np.sin(np.pi*x))*(np.cos(np.pi*y))
    return z

x=list(np.linspace(0,2,350))
y=list(np.linspace(0,2,350))


sec_deri=np.zeros((350,350))
fval=np.zeros((350,350))
for i in range(350):
    for j in range(350):
        sec_deri[i,j]=deri2(x[j],y[i])
        fval[i,j]=func(x[j],y[i])

xx=list(np.linspace(0,2,500))
yy=list(np.linspace(0,2,500))
yval=np.zeros((350,500))
yfinal=np.zeros((500,500))
for i in range(350):
    yval[i,:]=s.splintn(x,fval[i,:],sec_deri[i,:],xx)

ftrue=np.zeros((350,500))

for i in range(350):
    for j in range(500):
        ftrue[i,j]=func(xx[j],y[i])

for i in range(500):
    y2a=s.spline(y,yval[:,i],1.e40,1.e40)
    yfinal[i,:]=s.splintn(y,yval[:,i],y2a,yy)

error=np.zeros((500,500))
for i in range(500):
    for j in range(500):
        error[i,j]=abs(yfinal[i,j]-(func(xx[i],yy[j])))

x=np.arange(0,2,0.004)
y=np.arange(0,2,0.004)
x,y=np.meshgrid(x,y)
z=yfinal
z1=error
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y,np.log(z1), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("Cubic spline 3D-error-350pts")
