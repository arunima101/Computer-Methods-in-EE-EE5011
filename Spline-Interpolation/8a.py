from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.special as sp


def func(x,y):
    z=0.0
    z=(np.sin(np.pi*x))*(np.cos(np.pi*y))
    return z

def locate(x,var):
    jl=jm=jl=j=0
    
    ju=len(x)
    
    while((ju-jl)>1):
        jm= (ju+jl) >>1
        if(var>=x[jm]):
            jl=jm
        else:
            ju=jm
    if (var==x[0]):
         j=0
    if(var==x[-1]):
         j=len(x)-2
    else:
        j=jl
    return j


x=list(np.linspace(0,2,350))
y=list(np.linspace(0,2,350))
actual=np.zeros((500,500))

xx=list(np.linspace(0,2,500))
yy=list(np.linspace(0,2,500))

for i in range(len(xx)):
    for j in range(len(yy)):
        actual[i,j]=func(xx[i],yy[j])


interpol=np.zeros((500,500))

den=1
for m in range(len(xx)):
    for n in range(len(yy)):
     i=locate(x,xx[m])
     j=locate(y,yy[n])
     den=(x[i]-x[i+1])*(y[j]-y[j+1])
     interpol[m,n]=((func(x[i],y[j]))*(xx[m]-x[i+1])*(yy[n]-y[j+1])/den) + ((func(x[i+1],y[j]))*(x[i]-xx[m])*(yy[n]-y[j+1])/den) + ((func(x[i],y[j+1]))*(xx[m]-x[i+1])*(y[j]-yy[n])/den) + ((func(x[i+1],y[j+1]))*(x[i]-xx[m])*(y[j]-yy[n])/den)


error=np.zeros((500,500))

for i in range(500):
    for j in range(500):
        error[i,j]=abs(actual[i,j]-interpol[i,j])

x=np.arange(0,2,0.004)
y=np.arange(0,2,0.004)
x,y=np.meshgrid(x,y)
z=interpol
z1=error
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("3dplot-error-350pts")   


