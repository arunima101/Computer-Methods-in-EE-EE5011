import numpy as np
import scipy.special as sp
import scipy.integrate as s
import matplotlib.pyplot as plt
import romberg as r
import math as m

def efunc():
    return sp.jv(3,2.7)**2-sp.jv(4,2.7)*sp.jv(2,2.7)+abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2*(sp.kv(4,1.2)*sp.kv(2,1.2)-sp.kv(3,1.2)**2)

count=0
def func(u):
    global count
    count+=1
    f1=0
    if (u<1):
        f1=(sp.jv(3,2.7*u)**2)*u*2
    if (u>=1):
        f1=(2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u
    return f1

def trap3(func, a, b, n):
    if(n==1):
        return 0.5*(b-a)*(func(a)+func(b))
    else:
        d = (float)(b-a)/3**(n-1)
        sum=0.0
        x=a+d
        while(x<b):
            sum+=func(x)*d
            x+=d
        sum+=0.5*d*(func(a)+func(b))
        return sum


xx=[]; yy=[];y=[];error=[]

c=[]

for i in range(1,9):
    #count=0
    xx.append((20.0/3**(i-1))**2)
    yy.append(trap3(func,0,1,i)+trap3(func,1,20,i))
    y.append(r.polint(xx,yy,0)[0])
    error.append(abs(r.polint(xx,yy,0)[1]))


#y,error=r.polint(xx,yy,0)

plt.clf()
plt.loglog(xx,error,"r")
plt.xlabel("order of Decreasing h(right to left)")
plt.ylabel("Order of error")
plt.grid()
plt.savefig("12")


