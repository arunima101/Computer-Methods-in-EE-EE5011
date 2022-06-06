import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

def s(x,N):
    y=0.0
    for i in range(N+1):
        y=y+(1/(1+i))*sp.jv(i,x)
    return y

def alpha(x,i):
    return(2*(i+1)/x)

def beta():
    return -1

def c(i):
    return (1/(1+i))

def F(i,x):
    if (i==0):
        return sp.jv(0,x)
    if (i==1):
        return sp.jv(1,x)



'''
n=40
x=1.5
y=[0]*(n+2)
y[-1]=y[-2]=0
for k in range(n-1,-1,-1):
    y[k]=alpha(x,k)*y[k+1] + beta()*y[k+2] + c(k)

def F(i,x):
    if (i==0):
        return sp.jv(0,x)
    if (i==1):
        return sp.jv(1,x)


func=beta()*F(0,x)*y[1] + F(1,x)*y[0] + F(0,x)*c(0)
err=abs(func-s(x,40))
'''
func=[]
x=1.5
for n in range(2,41):
    y=[0]*(n+2)
    y[-1]=y[-2]=0
    for k in range(n-1,-1,-1):
        y[k]=alpha(x,k)*y[k+1] + beta()*y[k+2] + c(k)
    func.append(beta()*F(0,x)*y[1] + F(1,x)*y[0] + F(0,x)*c(0))
   
error=[]
for i in range(2,41):
    error.append(abs(s(x,i)-func[i-2]))

xval=[i for i in range (2,41)]
plt.clf()
plt.semilogy(xval,error,'r')
plt.savefig("clenshaw")
