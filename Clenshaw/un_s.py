import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

def s(x,N):
    y=0.0
    for i in range(N+1):
        y=y+(1/(1+i))*sp.jv(i,x)
    return y

# Forward recursion
x=15
j=[0,sp.jv(0,x) ]
for i in range(41):
    j.append(((2*(i))/x)*j[-1]-j[-2])

sum=0
error=[]
for n in range(2,41):
    for i in range(n):
        sum=sum+j[i]/(1+i)
    error.append(abs(s(x,n)-sum))

xval=[i for i in range(2,41)]
plt.clf()
plt.semilogy(xval,error,'c')
plt.xlabel("n Values")
plt.ylabel(" Order of error")
plt.title("Error for x=15")
plt.grid()

plt.savefig("Forward_1p5tttt")

# forward recursion ends

# Backward recursion

x=15
vals1p5=[0,1] #J61, J60
for i in range(60):
	vals1p5.append(2*(60-i)*vals1p5[-1]/x-vals1p5[-2])


for i in range(len(vals1p5)):
    vals1p5[i]=vals1p5[i]/vals1p5[-1]

vals1p5=vals1p5[::-1][:41]
j2=vals1p5
error1=[]
sum1=0
for n in range(41):
    sum1=sum1+ j2[n]/(1+n)
    error1.append(abs(s(x,n)-sum1))
    
xval=[i for i in range(41)]
plt.clf()
plt.plot(xval,error1,'g')
plt.xlabel("n Values")
plt.ylabel("Error")
plt.title("Error for x=15, Backward recursion")
plt.savefig('trial1p5')# need changes

#Backward ends





