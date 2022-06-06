import matplotlib.pyplot as plt
import math as m
from matplotlib import colors
import numpy as np

import polint as p


def locate(x,var,n):
    jl=jm=jl=diff=j=n1=0
    
    ju=len(x)

    d=[0]*(n+1)
    
    while((ju-jl)>1):
        jm= (ju+jl) >>1
        if(var>=x[jm]):
            jl=jm
        else:
            ju=jm
    if (var==x[0]):
         j=0
    if(var==x[-1]):
         j=len(x)
    else:
        j=jl
    n1=n//2
    diff=j-n1
    if(diff<=0):
        d=x[0:(n+1)]
    elif(diff>0 and j<(30-n1-1)):
         d=x[diff:(n+1+diff)]
    else:
         d=x[(len(x)-n-1):-1]
    y=[m.sin(m.pi*w)/(m.sqrt(1-w**2)) for w in d]
    return d,y

xx=np.linspace(0.1,0.9,30) 
#upon increasing beyond 0.9 and plotting for 1.1 or 1.2 or 1.1 shows runtime warning. The plots dont change for the values mentioned.
#The funstion itself converges to a value mathematically, but the error from interpolation blows up.


f=np.sin(np.pi*xx)/(np.sqrt(1-xx**2))

plt.figure()
plt.plot(list(xx),list(f),'ro')
plt.title("Original function")
plt.ylabel("function values at samples")
plt.xlabel("sampled points")
plt.savefig("Q5_B.png")


xx1=list(np.linspace(0.1,0.9,1000))
yy=[0]*1000
err_30_1=[0]*1000
err_true=[0]*1000
max_err=[0]*18
max_err_true=[0]*18

for j in range(3,20,1):
    x=y=[0]*(j+1)

    for i in range(1000):
        

        x,y=locate(xx,xx1[i],j)
        yy[i]=p.polint(x,y,xx1[i])[0]
        err_30_1[i]=np.absolute(p.polint(x,y,xx1[i])[1])
        err_true[i]=np.absolute((np.sin(np.pi*xx1[i])/(np.sqrt(1-xx1[i]**2)))-yy[i])
    max_err[j-3]=max((err_30_1))
    max_err_true[j-3]=max((err_true))

del max_err[-1]
del max_err_true[-1]
plt.figure()
xval=list(np.linspace(3,20,17))
plt.semilogy(xval,max_err,'r')
plt.title("Max error plots for 5C ")
plt.xlabel("Order of interpolation")
plt.ylabel("Error")
plt.grid()
plt.savefig("Max error_Q5 30pts.png")

