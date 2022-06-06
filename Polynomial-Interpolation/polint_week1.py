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



x=[0.0,0.0,0.0,0.0,0.0]
y=[0.0,0.0,0.0,0.0,0.0]

for i in range(5):
    if (i!=0):
        x[i]=x[i-1]+0.25
    y[i]=m.sin(x[i]+(x[i])**2)

xx=np.linspace(-0.5,1.5,200)
yy0=np.sin(xx+xx**2)
yy=[p.polint(x,y,w)[0] for w in xx]
err=[p.polint(x,y,w)[1] for w in xx]
err1=np.absolute(np.array(err))
err2=abs(yy0-yy)

plt.clf()
plt.figure(1)
plt.plot(x,y,'ro')
plt.title("Table values")
plt.xlabel("Sample Values")
plt.ylabel("Function values")
plt.grid()
plt.savefig("week1_0.png")

plt.plot(xx,yy,'r')
plt.title("Interpolated values")
plt.xlabel("Sample Values")
plt.ylabel("Function values from interpolation")
plt.grid()
plt.savefig("week1_1.png")

plt.figure(2)

plt.plot(xx,yy0,'g')
plt.title("True values")
plt.xlabel("Sample Values")
plt.ylabel("True Function values")
plt.grid()
plt.savefig("week1_2.png")

plt.figure(3)
plt.clf()
plt.plot(xx,err1,'r')
plt.title("Error values")
plt.xlabel("Sample Values")
plt.ylabel("Error values from interpolation")
plt.grid()
plt.savefig("week1_3")

plt.figure(4)
plt.clf()
plt.semilogy(xx,err2,'g')
plt.title("Logarithmic error")
plt.xlabel("Sample Values")
plt.ylabel("Error values for interpolation")
plt.grid()
plt.savefig("week1_4")

x=[0]*5
y=[0]*5

x1=list(np.linspace(0,1,30))
y1=[0]*30
for i in range(30):
    y1[i]=m.sin(x1[i]+(x1[i])**2)
    

xx1=list(xx)
yy1=[0]*200
err_30_1=[0]*200
err_est=[0]*200
max_err=0
max_err1=[0]*18



for i in range(200):
    x,y=locate(x1,xx[i],5)
    yy1[i]=p.polint(x,y,xx1[i])[0]
    err_30_1[i]=p.polint(x,y,xx1[i])[1]
    err_est[i]=(np.sin(xx1[i]+(xx1[i])**2))-yy1[i]





plt.figure(5)
plt.semilogy(xx1,(np.absolute(err_30_1)),color="red",label='Estimated error')
plt.legend(loc='lower right')
plt.title(" Error values for 30 points for nth order")
plt.semilogy(xx1,(np.absolute(err_est)),color="green",label='True error')
plt.legend(loc='lower right')
plt.xlabel("Values")
plt.ylabel("Error values")
plt.grid()
plt.savefig("week1_30_5.png")
