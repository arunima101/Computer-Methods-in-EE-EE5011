from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

x=np.loadtxt("error_from_exact_formula.txt",usecols=0)
ya=np.loadtxt("error_from_exact_formula.txt",usecols=1)
ys=np.loadtxt("error_from_exact_formula.txt",usecols=2)
x=np.delete(x,0,0)
ya=np.delete(ya,0,0)
ys=np.delete(ys,0,0)

plt.clf()
x1,=plt.loglog(x,ya,'g',label="Non cancelling root")
y,=plt.loglog(x,ys,'r',label="Roots with near cancellation")
plt.legend(handles=[x1,y])
plt.xlabel("Alpha values")
plt.ylabel("Order of error")
plt.grid()
plt.savefig("Error from exact formula")

x1=np.loadtxt("error_from_accurate_formula.txt",usecols=0)
ya1=np.loadtxt("error_from_accurate_formula.txt",usecols=1)
ys1=np.loadtxt("error_from_accurate_formula.txt",usecols=2)
x1=np.delete(x1,0,0)
ya1=np.delete(ya1,0,0)
ys1=np.delete(ys1,0,0)

plt.clf()
m,=plt.loglog(x1,ya1,'g',label="Roots with  near cancellation")
n,=plt.loglog(x1,ys1,'r',label="Non cancelling root")
plt.legend(handles=[m,n])
plt.xlabel("Alpha values")
plt.ylabel("Order of error")
plt.grid()
plt.savefig("Error from accurate formula")

