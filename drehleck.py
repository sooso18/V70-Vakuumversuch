import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import sem
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
#Erste Messung
t,p1,p2,p3,p4,p5,p6= np.genfromtxt("content/leck_dreh.txt",unpack=True)
V=ufloat(34,3.4)
t0=np.column_stack([p1,p2,p3])
p=t0.mean(axis=1)
perr=0.1*p
perr1=3.6

np.savetxt("content/drehleck1tab.csv",np.column_stack([t,p,perr,p4,p5,p6]),delimiter=",",fmt=["%4.2f","%4.2f","%4.2f","%4.2f","%4.2f","%4.2f"])

def f(x, a, b):
    return a*x+b

params, covariance_matrix = curve_fit(f, t, p)
errors = np.sqrt(np.diag(covariance_matrix))

print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
a=ufloat(params[0],errors[0])
pg=ufloat(0.5,0.05)
s1=(V/pg)*a

plt.grid()
plt.errorbar(t,p,yerr=perr,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params), 'r-', label='Fit')
plt.xlabel(r't $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
#plt.show()
plt.clf()

#plot2

def f(x, a, b):
    return a*x+b

params1, covariance_matrix1 = curve_fit(f, t, p4)
errors1 = np.sqrt(np.diag(covariance_matrix1))

print('a=', params1[0], '+-', errors1[0])
print('b=', params1[1], '+-', errors1[1])
a1=ufloat(params1[0],errors1[0])
pg1=ufloat(10,0.1)
s2=(V/pg1)*a1
perr1=3.6

plt.grid()
plt.errorbar(t,p4,yerr=perr1,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params1), 'r-', label='Fit')
plt.xlabel(r't $[s]$$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
#plt.show()
plt.clf()

#plot 3
def f(x, a, b):
    return a*x+b

params2, covariance_matrix2 = curve_fit(f, t, p5)
errors2 = np.sqrt(np.diag(covariance_matrix2))

print('a=', params2[0], '+-', errors2[0])
print('b=', params2[1], '+-', errors2[1])
a2=ufloat(params2[0],errors2[0])
pg2=ufloat(50,3.6)
s3=(V/pg2)*a2
perr1=3.6

plt.grid()
plt.errorbar(t,p5,yerr=perr1,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params2), 'r-', label='Fit')
plt.xlabel(r't $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
#plt.show()
plt.clf()



#plot 4
def f(x, a, b):
    return a*x+b

params4, covariance_matrix4 = curve_fit(f, t, p6)
errors4 = np.sqrt(np.diag(covariance_matrix4))

print('a=', params4[0], '+-', errors4[0])
print('b=', params4[1], '+-', errors4[1])
a4=ufloat(params4[0],errors4[0])
pg4=ufloat(100,3.6)
s4=(V/pg4)*a4
perr1=3.6

plt.grid()
plt.errorbar(t,p6,yerr=perr1,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params4), 'r-', label='Fit')
plt.xlabel(r't  $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
#plt.show()
plt.clf()

print("S1 in l/s=",s1)
print("S2 in l/s=",s2)
print("S3 in l/s=",s3)
print("S4 in l/s=",s4)
