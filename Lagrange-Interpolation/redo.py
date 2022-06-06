from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import math

x=np.loadtxt("output.txt",usecols=0)
y=np.loadtxt("output.txt",usecols=1)
f=np.sin(x)
print(f)
error=np.absolute(f-y)

plt.plot(x,error)
plt.grid()
plt.savefig("redo.png")


x1=np.loadtxt("output1.txt",usecols=0)
y1=np.loadtxt("output1.txt",usecols=1)
f1=np.sin(x1)

error1=np.absolute(f1-y1)
plt.figure()

plt.plot(x1,error1)
plt.grid()
plt.savefig("redo3.png")
