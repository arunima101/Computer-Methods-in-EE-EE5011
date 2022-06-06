#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt 
import matplotlib
import math


# In[2]:


def func(x):
    return x*sp.jv(1,x)


# In[3]:


def chebft(a,b,n):
    bma=0.5*(b-a)
    bpa=0.5*(b+a)
    f=[]
    c=[]
    for k in range(1,n+1):
        y=np.cos(np.pi*(k-0.5)/n)
        f.append(func(y*bma+bpa))
    fac=2/n
    for j in range(n):
        summation=0
        for k in range(1,n+1):
            summation += f[k-1]*np.cos(np.pi*j*(k-0.5)/n)
        c.append(fac*summation)
    return c


# In[4]:


coeffients=chebft(0,5,50) 


# In[5]:


fig, ax = plt.subplots(figsize=(12, 12))
plt.grid(True)
ax.semilogy(range(50),np.abs(coeffients),'g')
ax.semilogy(range(50),np.abs(coeffients),'bo')
ax.set_xlabel("Number of points",fontsize=20)
ax.set_ylabel("Order of Error",fontsize=20)


# ### Approach 1

# In[6]:


def chebyshev_poly(n,x): #Calculating the Tn(x) by forward recursion
    if n==0:
        return 1
    if n==1:
        return x
    else:
        return 2*x*chebyshev_poly(n-1,x)-chebyshev_poly(n-2,x)
        


# In[7]:


def function_from_chebyshev(m,x,coeff,a,b):
    function=0
    y=-1+(x-a)*2/(b-a)
    for k in range(m):
         function+=coeff[k]*chebyshev_poly(k,y)
            
    function=function-(0.5*coeff[0])
    return function    


# In[8]:


def error1():
    error=[]
    x=np.linspace(0,5,20)
    for i in range(20):
        error.append(np.abs(func(x[i])-function_from_chebyshev(20,x[i],coeffients,0,5)))
    return error


# In[9]:


x=np.linspace(0,5,20)
fig, ax = plt.subplots(figsize=(10, 8))
plt.grid(True)
ax.semilogy(x,error1(),'r')
ax.semilogy(x,error1(),'bo')
ax.set_xlabel("Points in the Interval",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# ### Approach 2

# In[10]:


def mapper(x, min_x, max_x, min_to, max_to):
    return (x - min_x) / (max_x - min_x) * (max_to - min_to) + min_to


# In[11]:


def cheb_approx(x, m, a, b, coef):
    T_n_2 = 1
    T_n_1 = mapper(x, a, b, -1, 1)
    T_n = 0
    function = coef[0] / 2 + coef[1] * T_n_1
 
    x = 2 * T_n_1 
    i = 2
    while i < m:
        T_n = x * T_n_1 - T_n_2
        function = function + coef[i] * T_n
        (T_n_2, T_n_1) = (T_n_1, T_n)
        i += 1
 
    return function


# In[12]:


def error():
    error=[]
    x=np.linspace(0,5,20)
    for i in range(20):
        error.append(np.abs(func(x[i])-cheb_approx(x[i], 20, 0, 5, coeffients)))
    return error


# In[13]:



fig, ax = plt.subplots(figsize=(10, 8))
plt.grid(True)
ax.semilogy(x,error(),'m')
ax.semilogy(x,error(),'ko')
ax.set_xlabel("Points in the Interval",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[14]:


def function_derivative(x):
    return x*sp.jv(0,x)


# In[15]:


def chder(a,b,coeff,m):
    coeff_der=[0]*m
    coeff_der[-2]=2*(m-1)*coeff[m-1]
    j=m-3
    while j>=0:
        coeff_der[j]=coeff_der[j+2]+2*(j+1)*coeff[j+1]
        j-=1
    normalized_list=[value*2/(b-a) for value in coeff_der]
    return normalized_list
        


# In[16]:


derivative_coeff=chder(0,5,coeffients,20)


# In[17]:


def error_derivative():
    error=[]
    x=np.linspace(0,5,20)
    for i in range(20):
        error.append(np.abs(function_derivative(x[i])-function_from_chebyshev(20,x[i],derivative_coeff,0,5)))
    return error


# In[18]:


x=np.linspace(0,5,20)
fig, ax = plt.subplots(figsize=(10, 8))
plt.grid(True)
ax.semilogy(x,error_derivative(),'-+')
ax.semilogy(x,error_derivative(),'ko')
ax.set_xlabel("Points in the Interval",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error in the interval for f'(x)-Chebyshev",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[19]:


def function_primitive_der(x,d):
    deri=(func(x+d/2)-func(x-d/2))/d
    return deri


# In[20]:


def error_primitive_deri(a,b,d):
    x=np.linspace(a,b,20)
    error=[]
    for i in range(20):
        error.append(np.abs(function_derivative(x[i])-function_primitive_der(x[i],d)))
    return error
    


# In[21]:


x=np.linspace(0,5,20)
fig,ax = plt.subplots(figsize=(10, 8))
plt.grid(True)
ax.semilogy(x,error_primitive_deri(0,5,10**(-6)),'b')
ax.semilogy(x,error_primitive_deri(0,5,10**(-5)),'g')
#ax.semilogy(x,error_primitive_deri(0,5,10**(-6)),'ko')
ax.semilogy(x,error_primitive_deri(0,5,0.001),'k')
#ax.semilogy(x,error_primitive_deri(0,5,0.001),'bo')
ax.set_xlabel("Points in the Interval",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error in the interval for f'(x)-Diffrence of samples",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(["del=10^-6","del=10^-5","del=10^-3"])
plt.rcParams['legend.fontsize'] = 15


# In[22]:


def get_d():
    return 3

def f(x):
    return math.exp(x)

def g(x):
    d=get_d()
    return 1/(x**2+ d**2)

def h(x):
    d=get_d()
    return 1/((math.cos(math.pi*x/2))**2+d**2)

def u(x):
    return math.exp(-1*abs(x))

def v(x):
    return math.sqrt(x+1.1)
    


# In[23]:


def chebft_modified(a,b,n,func):
    bma=0.5*(b-a)
    bpa=0.5*(b+a)
    f=[]
    c=[]

    for k in range(1,n+1):
        y=np.cos(np.pi*(k-0.5)/n)
        f.append(func(y*bma+bpa))
    fac=2/n
    for j in range(n):
        summation=0
        for k in range(1,n+1):
            summation += f[k-1]*np.cos(np.pi*j*(k-0.5)/n)
        c.append(fac*summation)
    return c


# In[24]:


#For f(x),g(x,d),h(x,d),v(x)

coeff_f=chebft_modified(-1,1,100,f)
coeff_v=chebft_modified(-1,1,100,v)
coeff_g=chebft_modified(-1,1,100,g)
coeff_h=chebft_modified(-1,1,100,h)

fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(range(100),np.abs(coeff_f),'g')
ax.semilogy(range(100),np.abs(coeff_g),'r')
ax.semilogy(range(100),np.abs(coeff_v),'k')
ax.semilogy(range(100),np.abs(coeff_h),'b')
ax.set_xlabel("Number of points",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Coeffecients",fontsize=15)
plt.legend(["f(x)","g(x)","v(x)","h(x)"])
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[25]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)

ax.semilogy(range(100),np.abs(coeff_g),'r')
ax.semilogy(range(100),np.abs(coeff_h),'b')
ax.set_xlabel("Number of points",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Coeffecients,delta=3",fontsize=15)
plt.legend(["g(x)","h(x)"])
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[26]:


def error_function(a,b,m,coeff,func):
    error=[]
    x=np.linspace(a,b,m)
    for i in range(len(x)):
        error.append(np.abs(func(x[i])-cheb_approx(x[i], m, a, b, coeff)))
    return error


# In[27]:


# For h(x) cutoff is 30 coeffients

error_h=error_function(-1,1,30,coeff_h,h) 


# In[28]:


# For f(x) the cutoff is 18 coeffients

error_f=error_function(-1,1,18,coeff_f,f)


# In[29]:


# For g(x) the cutoff if 20 coeffients

error_g=error_function(-1,1,20,coeff_g,g)


# In[30]:


#For v(x) the cutoff is 60 coeffients

error_v=error_function(-1,1,60,coeff_v,v)


# In[31]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(np.linspace(-1,1,30),error_h,'m')
ax.semilogy(np.linspace(-1,1,30),error_h,'ko')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for h(x)-Chebyshev Fit",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[32]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(np.linspace(-1,1,18),error_f,'g')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for f(x)-Chebyshev Fit",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[33]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(np.linspace(-1,1,20),error_g,'--b')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for g(x)-Chebyshev Fit",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[34]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(np.linspace(-1,1,60),error_v,'k')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for v(x)-Chebyshev Fit-60 coefficients",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# ### For u(x)

# In[35]:


coeff_u=chebft_modified(-1,1,100,u)


# In[36]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(range(100),np.abs(coeff_u),'k')
ax.set_xlabel("Number of points",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Coeffients for u(x)",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[37]:


error_u=error_function(-1,1,50,coeff_u,u)


# In[38]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
ax.semilogy(np.linspace(-1,1,50),error_u,'r')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for u(x)-with whole range",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[39]:


# Breaking the range for u(x)

coeff_u1=chebft_modified(-1,0,50,u)
error_u1=error_function(-1,0,25,coeff_u1,u)

coeff_u2=chebft_modified(0,1,50,u)
error_u2=error_function(0,1,25,coeff_u2,u)


# In[40]:


error_u1.extend(error_u2)


# In[41]:


fig, ax = plt.subplots(figsize=(12,12))
plt.grid(True)
plt.semilogy(np.linspace(-1,1,50),error_u1,'b')
plt.semilogy(np.linspace(-1,1,50),error_u1,'ro')
ax.set_xlabel("-1<x<1",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for u(x)-with range -1<x<0 and 0<x<1",fontsize=15)
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# ### Fourier 

# In[46]:


from scipy.fft import fft, ifft, fftshift,ifftshift
from scipy.integrate import quad
import scipy as sc


# In[73]:


def fouriercoeffs(func,N):
    fcoeff=[]
    for i in range(N):
        fcoeff.append(quad(func,-1,1,args=(i))[0])
    return np.array(fcoeff)

def fouriercoeffs_u(func,N):
    fcoeff1=[]
    fcoeff2=[]
    N=N//2
    for i in range(N):
        fcoeff1.append(quad(func,-1,0,args=(i))[0])
        fcoeff2.append(quad(func,0,1,args=(i))[0])
    fcoeff1.extend(fcoeff2)
    return np.array(fcoeff1)


# In[48]:


def f1(x,m):
    return f(x)*np.cos(m*(x+1)*np.pi/2)


# In[49]:


def h1(x,m):
    return h(x)*np.cos(m*(x+1)*np.pi/2)


# In[50]:


def g1(x,m):
    return g(x)*np.cos(m*(x+1)*np.pi/2)


# In[51]:


def u1(x,m):
    return u(x)*np.cos(m*(x+1)*np.pi/2)


# In[52]:


def v1(x,m):
    return v(x)*np.cos(m*(x+1)*np.pi/2)


# In[76]:


coeff_fourier_f=fouriercoeffs(f1,100)
coeff_fourier_h=fouriercoeffs(h1,100)
coeff_fourier_g=fouriercoeffs(g1,100)
coeff_fourier_u=fouriercoeffs_u(u1,100)
coeff_fourier_v=fouriercoeffs(v1,100)
coeff_u1.extend(coeff_u2)


# In[84]:


fig, ax = plt.subplots(figsize=(12,12))
plt.semilogy(range(100),abs(coeff_fourier_h),'b')
plt.semilogy(range(100),coeff_h,'g')
plt.grid(True)
ax.set_xlabel("No of coefficients",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Coefficients for h(x): Fourier and Chebyshev",fontsize=15)
plt.legend(["Fourier Coefficients","Chebyshev Coefficients"])
plt.rcParams['legend.fontsize'] = 15
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# ### I also tried to implement chebyshev fittiing to that fuction in romberg assignment. There was a huge spike at x=1, so the fitting is quiet nice from (0,1) but the errors are poor from there onwards.

# In[85]:


def new_func(u):
    f1=0
    if (u<1):
        f1=(sp.jv(3,2.7*u)**2)*u*2
    if (u>=1):
        f1=(2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u
    return f1


# In[86]:


new_coeff1=chebft_modified(0,1,200,new_func)
new_coeff2=chebft_modified(1,15,200,new_func)


# In[87]:


err1=error_modified(0,1,50,new_coeff1,new_func)
err2=error_modified(1,15,50,new_coeff1,new_func)


# In[88]:


x=np.linspace(0,15,50)
fig, ax = plt.subplots(figsize=(12,12))
plt.semilogy(range(50),err1,'k')
plt.grid(True)


# In[163]:


x=np.linspace(0,15,50)
fig, ax = plt.subplots(figsize=(12,12))
plt.semilogy(range(50),err2,'k')
plt.grid(True)


# ### Using Clenshaw

# In[117]:


def eval(a,b,x,c):
        
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (c[-1], 0)             # Special case first step for efficiency
        for cj in c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * c[0] 
    
def eval_fourier(a,b,x,c):
        
        assert(a <= x <= b)
        #y = (2.0 * x - a - b) * (1.0 / (b - a))
        #y2 = 2.0 * y
        y2=x
        y=x
        (d, dd) = (c[-1], 0)             # Special case first step for efficiency
        for cj in c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * c[0]     


# In[118]:


def error_modified(a,b,m,coeff,func):
    error=[]
    x=np.linspace(a,b,m)
    for i in range(len(x)):
        error.append(np.abs(func(x[i])-eval(a, b,x[i], coeff[:m])))
    return error
def error_modified_fourier(a,b,m,coeff,func):
    error=[]
    x=np.linspace(a,b,m)
    for i in range(len(x)):
        error.append(np.abs(func(x[i])-eval_fourier(a, b,x[i], coeff[:m])))
    return error


# In[155]:


#Using Clenshaw

error_cheby_f=error_modified(-1,1,18,coeff_f,f)
error_fourier_f=error_modified_fourier(-1,1,18,coeff_fourier_f,f)

error_cheby_g=error_modified(-1,1,30,coeff_g,g)
error_fourier_g=error_modified_fourier(-1,1,30,coeff_fourier_g,g)

error_cheby_h=error_modified(-1,1,30,coeff_h,h)
error_fourier_h=error_modified_fourier(-1,1,30,coeff_fourier_h,h)

error_cheby_v=error_modified(-1,1,60,coeff_v,v)
error_fourier_v=error_modified_fourier(-1,1,60,coeff_fourier_v,v)



coeff_u1_clenshaw=chebft_modified(-1,0,50,u)
error_u1_clenshaw=error_modified(-1,0,25,coeff_u1_clenshaw,u)

coeff_u2_clenshaw=chebft_modified(0,1,50,u)
error_u2_clenshaw=error_function(0,1,25,coeff_u2_clenshaw,u)
error_u1_clenshaw.extend(error_u2_clenshaw)


# In[161]:


error_fourier_u1=error_modified_fourier(-1,0,25,coeff_fourier_u[:30],u)
error_fourier_u2=error_modified_fourier(0,1,25,coeff_fourier_u[20:],u)
error_fourier_u1.extend(error_fourier_u2)


# In[159]:


len(error_u1_clenshaw)


# In[143]:


# Not using Clenshaw
def error_nclenshaw(function,coeffients,a,b,m):
    error=[]
    x=np.linspace(a,b,m)
    for i in range(m):
        error.append(np.abs(function(x[i])-cheb_approx(x[i], m, a, b, coeffients)))
    return error

err_clenshaw_f=error_nclenshaw(f,coeff_f,-1,1,18)
err_clenshaw_g=error_nclenshaw(g,coeff_g,-1,1,30)
err_clenshaw_h=error_nclenshaw(h,coeff_h,-1,1,30)
err_clenshaw_v=error_nclenshaw(h,coeff_v,-1,1,60)


# ### Using and not using clenshaw doesnt show much diffrence in case of chebyshev. But Clenshaw for fourier approximation gives too much error.

# In[162]:


fig, ax = plt.subplots(figsize=(12,12))
plt.semilogy(range(50),error_u1_clenshaw,'g')
plt.semilogy(range(50),error_fourier_u1,'r')
plt.grid(True)
ax.set_xlabel("Number of coefficients",fontsize=15)
ax.set_ylabel("Order of Error",fontsize=15)
ax.set_title("Error for u(x)-With clenshaw",fontsize=15)
plt.legend(["Chebyshev error","Fourier error"])
plt.rcParams['legend.fontsize'] = 25
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# In[ ]:




