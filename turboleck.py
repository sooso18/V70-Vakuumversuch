import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import sem
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
#Erste Messung
t,p1,p2,p3,p4= np.genfromtxt("content/leck_turbo.txt",unpack=True)
V=ufloat(33,3.3)
p1=p1*1e-4
p2=p2*1e-4
p3=p3*1e-4
p4=p4*1e-4
perr=0.3*p1
perr2=0.3*p2
perr3=0.3*p3
perr4=0.3*p4

np.savetxt("content/turboleck1tab.csv",np.column_stack([t,p1,perr,p2,perr2,p3,perr3,p4,perr4]),delimiter=",",fmt=["%4.2f","%4.2f","%4.2f","%4.2f","%4.2f","%4.2f","%4.2f","%4.2f","%4.2f"])

def f(x, a, b):
    return a*x+b

params, covariance_matrix = curve_fit(f, t, p1)
errors = np.sqrt(np.diag(covariance_matrix))

print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
a=ufloat(params[0],errors[0])
pg=ufloat(5e-5,1.5e-5)
s1=(V/pg)*a

plt.grid()
plt.errorbar(t,p1,yerr=perr,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params), 'r-', label='Fit')
plt.xlabel(r't $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.show()
plt.clf()

#plot2

def f(x, a, b):
    return a*x+b

params1, covariance_matrix1 = curve_fit(f, t, p2)
errors1 = np.sqrt(np.diag(covariance_matrix1))

print('a=', params1[0], '+-', errors1[0])
print('b=', params1[1], '+-', errors1[1])
a1=ufloat(params1[0],errors1[0])
pg1=ufloat(7e-5,2.1e-5)
s2=(V/pg1)*a1


plt.grid()
plt.errorbar(t,p2,yerr=perr2,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params1), 'r-', label='Fit')
plt.xlabel(r't $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.show()
plt.clf()

#plot 3
def f(x, a, b):
    return a*x+b

params2, covariance_matrix2 = curve_fit(f, t, p3)
errors2 = np.sqrt(np.diag(covariance_matrix2))

print('a=', params2[0], '+-', errors2[0])
print('b=', params2[1], '+-', errors2[1])
a2=ufloat(params2[0],errors2[0])
pg2=ufloat(1e-4,3e-5)
s3=(V/pg2)*a2


plt.grid()
plt.errorbar(t,p3,yerr=perr3,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params2), 'r-', label='Fit')
plt.xlabel(r't $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.show()
plt.clf()



#plot 4
def f(x, a, b):
    return a*x+b

params4, covariance_matrix4 = curve_fit(f, t, p4)
errors4 = np.sqrt(np.diag(covariance_matrix4))

print('a=', params4[0], '+-', errors4[0])
print('b=', params4[1], '+-', errors4[1])
a4=ufloat(params4[0],errors4[0])
pg4=ufloat(2e-4,6e-5)
s4=(V/pg4)*a4


plt.grid()
plt.errorbar(t,p4,yerr=perr4,fmt=".",color="k",label="Messdaten")
plt.plot(t,f(t,*params4), 'r-', label='Fit')
plt.xlabel(r't  $[s]$')
plt.ylabel(r'p(t) $[mbar]$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.show()
plt.clf()

print("S1 in l/s=",s1)
print("S2 in l/s=",s2)
print("S3 in l/s=",s3)
print("S4 in l/s=",s4)
