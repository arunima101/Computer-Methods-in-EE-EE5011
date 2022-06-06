from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import math


#all information for 8th Order interpolation
x=np.loadtxt("output.txt",usecols=0)
y=np.loadtxt("output.txt",usecols=1)

f=np.sin(x)

error=np.absolute(f-y)
f3=np.delete(f,0,0)
y3=np.delete(y,0,0)
x3=np.delete(x,0,0)
val=np.log10(np.absolute(f3-y3))
plt.plot(x3,val,color="green",label='8th Order')

#cubic
x1=np.loadtxt("output1.txt",usecols=0)
y1=np.loadtxt("output1.txt",usecols=1)
f1=np.sin(x1)

error1=np.absolute(f1-y1)
f2=np.delete(f1,0,0)
y2=np.delete(y1,0,0)
x2=np.delete(x1,0,0)

val=np.log10(np.absolute(f2-y2))
plt.plot(x2,val,color="blue",label="3rd Order")
plt.legend()
plt.title("Logarithmic error")
plt.savefig("Plot_log_error.png")


#Plotting for 8th order
figure,axis=plt.subplots(3)

axis[0].plot(x,y)
axis[0].set_title("Interpolated function for 8th Order")


axis[1].plot(x,f)
axis[1].set_title("Original Sine function")


axis[2].plot(x,error)
axis[2].set_title("Error Vs X in radian for 8th Order")
figure.tight_layout()

plt.savefig("Plot_8.png")


#Plotting for cubic interpolation
figure,axis1=plt.subplots(3)

axis1[0].plot(x1,y1)
axis1[0].set_title("Interpolated function for cubic")

axis1[1].plot(x1,f1)
axis1[1].set_title("Original Sine function")

axis1[2].plot(x1,error1)
axis1[2].set_title("Error Vs X in radian for cubic")
figure.tight_layout()
plt.savefig("Plot_3.png")











