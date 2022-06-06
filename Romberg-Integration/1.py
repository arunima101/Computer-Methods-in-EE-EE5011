import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

def func(u):
    f1=0
    if (u<1):
        f1=(sp.jv(3,2.7*u)**2)*u*2
    if (u>=1):
        f1=(2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u
    return f1

val=[]
xval=list(np.logspace(-4,0,1000))

for i in xval:
    val.append(func(i))

plt.clf()
plt.loglog(xval,val,'g')
plt.grid()
plt.title("Semi-Log plot for function")
plt.xlabel("Values of u")
plt.ylabel("Function values")
plt.savefig("Function_semilog")
plt.clf()
plt.plot(xval,val,'g')
plt.title("Plot for function")
plt.xlabel("Values of u")
plt.ylabel("Function values")
plt.grid()
plt.savefig("Function")