import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import scipy.integrate as s

def func(u):
    f1=0
    if (u<1):
        f1=2*u*(sp.jv(3,2.7*u))**2
    if (u>=1):
        f1=(2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u
    return f1

def efunc():
    return sp.jv(3,2.7)**2-sp.jv(4,2.7)*sp.jv(2,2.7)+abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2*(sp.kv(4,1.2)*sp.kv(2,1.2)-sp.kv(3,1.2)**2)

out=s.quad(func,0,20,full_output=0)
err=out[0]-efunc()

error=[]
for i in range(10,30):
    out=s.quad(func,0,i,full_output=0)
    error.append(abs(out[1]))

plt.clf()
plt.semilogy(range(10,30),error,'c')
plt.xlabel("Values of upper bound")
plt.ylabel("Order of error")
plt.grid()
plt.savefig("4")
    
