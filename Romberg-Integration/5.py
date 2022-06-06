import numpy as np
import scipy.special as sp
import scipy.integrate as s
import matplotlib.pyplot as plt
import romberg as r

def efunc():
    return sp.jv(3,2.7)**2-sp.jv(4,2.7)*sp.jv(2,2.7)+abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2*(sp.kv(4,1.2)*sp.kv(2,1.2)-sp.kv(3,1.2)**2)

count =0
def func(u):
    global count
    count+=1
    f1=0
    if (u<1):
        f1=(sp.jv(3,2.7*u)**2)*u*2
    if (u>=1):
        f1=(2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u
    return f1

s1=0
c=[]
err=[]
for i in range(1,20):
    
    s1=r.trapzd(func,0,20,s1,i)
    print (i,s1,s1-efunc())
    err.append(abs(s1-efunc()))
    c.append(count)

plt.clf()
plt.loglog(c,err,'m')
plt.grid()
plt.xlabel("No. of function Calls")
plt.ylabel("Error in Trapeziodal")
plt.savefig("Trapeziodal-err")


   
