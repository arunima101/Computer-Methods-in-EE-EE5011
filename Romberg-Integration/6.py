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



error=[]
call=[]
value=[]

for i in range(-1,-14,-1):
    value.append(r.qromb(func,0,20,10**i)[0])
    error.append(abs(r.qromb(func,0,20,10**i)[1]))
    call.append(r.qromb(func,0,20,10**i)[2])

plt.clf()
plt.loglog(call,error,'r')
plt.xlabel("Number of calls")
plt.ylabel("Order of error")
plt.grid()
plt.savefig(' Error in qromb')




  

