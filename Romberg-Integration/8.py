import numpy as np
import scipy.special as sp
import scipy.integrate as s
import matplotlib.pyplot as plt
import romberg as r

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

def qrombp(f,a,b):
    out=error=0
    k=5
    xx=yy=[]
    for i in range(1,k+1):
        out=r.trapzd(f,a,b,out,i)
        xx.append(((b-a)/(2**(i-1)))**2)
        yy.append(out)
    out,error=r.polint(xx,yy,0)
    return out,error
 
out,error=qrombp(func,0,20)
c=[]
x=[]
for i in range(3,21):
    count=0
    x.append(r.qromb(func,0,20,1e-8,i)[1])
    c.append(count)

plt.clf()
plt.semilogy(range(3,21),c,'c')
plt.xlabel("order from qromb")
plt.ylabel("Number of function calls")
plt.grid()
plt.savefig("9")



