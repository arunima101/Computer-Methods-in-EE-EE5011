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
       

    y=[m.sin(w+w**2) for w in d]
    return d,y


xx=np.linspace(0,1,200)

x1=list(np.linspace(0,1,30))


xx1=list(xx)
yy1=[0]*200
err_30_1=[0]*200
err_est=[0]*200
max_err=0
max_err1=[0]*18
max_err_est=[0]*18



for j in range(3,20,1):
    x=y=[0]*(j+1)

    for i in range(200):
        

        x,y=locate(x1,xx[i],j)
        yy1[i]=p.polint(x,y,xx1[i])[0]
        err_30_1[i]=np.absolute(p.polint(x,y,xx1[i])[1])
        err_est[i]=np.absolute((np.sin(xx1[i]+(xx1[i])**2))-yy1[i])
    max_err1[j-3]=max((err_30_1))
    max_err_est[j-3]=max((err_est))
    

del max_err1[-1]
del max_err_est[-1]


xval=np.linspace(3,20,17)
yval1=np.log10(max_err_est)
yval2=np.log10(max_err1)
plt.clf()
plt.semilogy(xval,max_err_est,'g',label='True error')
plt.legend(loc='upper right')
plt.semilogy(xval,max_err1,'r',label="Estimated error")
plt.legend(loc='upper right')
plt.title("Max error plots")
plt.xlabel("Order of interpolation")
plt.ylabel("Error values")
plt.grid()
plt.savefig("Max errorQ404.png")



