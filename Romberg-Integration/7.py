import numpy as np
import scipy.special as sp
import scipy.integrate as s
import matplotlib.pyplot as plt
import romberg as r

def efunc():
    return sp.jv(3,2.7)**2-sp.jv(4,2.7)*sp.jv(2,2.7)+abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2*(sp.kv(4,1.2)*sp.kv(2,1.2)-sp.kv(3,1.2)**2)

count1=0
count2=0
def func1(u):
    global count1
    count1+=1
    return (sp.jv(3,2.7*u)**2)*u*2

def func2(u):
    global count2
    count2+=1
    return (2*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2)*((sp.kv(3,1.2*u))**2)*u

def q():
   out1=s.quad(func1,0,1,full_output=0)[0]
   out2=s.quad(func2,1,20,full_output=0)[0]
   result=out1+out2
   error=result-efunc()   # nevals=147+21=168!
   return result, error

def t():
    err=[]
    c=[]
    s1=s2=0
    for i in range(1,20):
    
       s1=r.trapzd(func1,0,1,s1,i)
       s2=r.trapzd(func2,1,20,s2,i)
       err.append(abs((s1+s2)-efunc()))
       c.append(count1+count2)
    return s1+s2,err,c

def qr():
    error=[]
    call1=[]
    value1=value2=0

    for i in range(-1,-11,-1):
       value1=(r.qromb(func1,1,20,10**i)[0])
       #value2=(r.qromb(func2,1,20,10**i)[0])
       #error.append(abs((value1+value2)-efunc()))
       error.append(abs((r.qromb(func1,1,20,10**i)[1])))
       call1.append(r.qromb(func1,1,20,10**i)[2])

    return value1,error,call1



result_q,error_q=q()

result_t,error_t,numcalls=t()

result_qr,error_qr,numcalls_qr=qr()




plt.clf()
plt.loglog(numcalls_qr,error_qr,"m")
plt.ylabel("Order of error")
plt.xlabel("Order of Number of calls")
plt.grid()
plt.savefig("Error in qromb with split1111")
