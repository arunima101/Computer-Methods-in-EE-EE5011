import numpy as np
import scipy.special as sp
import scipy.integrate as s
import matplotlib.pyplot as plt
import romberg as r
import scipy.interpolate as si

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

y=np.vectorize(func)
error=[]
error1=[]
c1=[]
c2=[]

for i in range(3,20):
    x=np.linspace(0,20,2**i)
    tck=si.splrep(x,y(x))
    out=si.splint(0,20,tck)
    error.append(abs(out-efunc()))
    c1.append(count)


for i in range(3,20):
    x1=np.linspace(0,1,2**(i-1))
    x2=np.linspace(1,20,2**(i-1))
    tck1=si.splrep(x1,y(x1))
    tck2=si.splrep(x2,y(x2))
    out=si.splint(0,1,tck1) + si.splint(1,20,tck2)
    error1.append(abs(out-efunc()))
    c2.append(count/2)

plt.clf()
plt.semilogy(range(3,20),error,'r',label="Without split")
plt.semilogy(range(3,20),error1,'g',label="With split, (0,1) and (1,20)")
plt.legend(loc="upper right")
plt.xlabel("log2(Number of points)")
plt.ylabel("Order of error")
plt.grid()
plt.savefig("101n")


