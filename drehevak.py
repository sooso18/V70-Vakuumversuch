import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import sem
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

p,ln1,abw1,p_err,ln2,abw2,t = np.genfromtxt('content/drehevaktab.txt',unpack=True ,delimiter=",")
##Drehschieberpumpe: Volumen 34L +- 3.4L (10%)
V=ufloat(34,3.4)
##Enddruck pe = 0.016 mbar mit Fehler 120mBar
pe=ufloat(1.6e-2,1.6e-3)
##Anfangsdruck p0 = 1000mbar (über 10mBar 0.3%)
p0=ufloat(1000,0.003*1200)
##nur für mittelwertbildung
    ##t0=np.column_stack([t1,t2,t3,t4,t5])
    ##t=t0.mean(axis=1)
    ##terr=sem(t,axis=1)???????

#perr1=3.6
#pges1=unp.uarray(p,perr1)
#pln1=(pges1-pe)/(p0-pe)
#lnpges1=unp.log(pln1)

#np.savetxt("content/drehevaktab.csv",np.column_stack([p,perr1,unp.nominal_values(lnpges1),unp.std_devs(lnpges1),t]),delimiter=",",fmt=["%3.2f","%3.2f","%3.2f","%3.2f","%4.2f"])

plt.grid()
plt.errorbar(t[:14], ln1[:14],yerr=abw1[:14],fmt='.',color="r",label="Messdaten")


def f(x, a, b):
    return a*x+b

params1, covariance_matrix1 = curve_fit(f, t[:14], ln1[:14])
errors1 = np.sqrt(np.diag(covariance_matrix1))

print('a1=', params1[0], '+-', errors1[0])
print('b1=', params1[1], '+-', errors1[1])

a1=ufloat(params1[0],errors1[0])
s1=-a1*V


#perr2=0.1*p
#pges2=unp.uarray(p,perr2)
#pln2=(pges2-pe)/(p0-pe)
#lnpges2=unp.log(pln2)

#np.savetxt("content/dreh_evaktab.csv",np.column_stack([p,perr2,unp.nominal_values(lnpges2),unp.std_devs(lnpges2),t]),delimiter=",",fmt=["%3.2f","%3.2f","%3.2f","%3.2f","%4.2f"])

plt.grid()
plt.errorbar(t[14:23], ln2[14:23] ,yerr=abw2[14:23] , fmt=".",color="b",label="Messdaten") 

def f(x, a, b):
    return a*x+b

params2, covariance_matrix2 = curve_fit(f, t[14:23], ln2[14:23])
errors2 = np.sqrt(np.diag(covariance_matrix2))

print('a2=', params2[0], '+-', errors2[0])
print('b2=', params2[1], '+-', errors2[1])
a2=ufloat(params2[0],errors2[0])
s2=-a2*V

plt.grid()
plt.errorbar(t[23:], ln2[23:] ,yerr=abw2[23:] , fmt=".",color="k",label="Messdaten") 

def f(x, a, b):
    return a*x+b

params3, covariance_matrix3 = curve_fit(f, t[22:], ln2[22:])
errors3 = np.sqrt(np.diag(covariance_matrix2))

print('a2=', params3[0], '+-', errors2[0])
print('b2=', params3[1], '+-', errors2[1])
a3=ufloat(params3[0],errors2[0])
s3=-a3*V

print("S1 in l/s =",s1)
print("S2 in l/s =",s2)
print("S3 in l/s =",s3)

plt.plot(t[:14], f(t[:14], *params1), 'r-', label='Fit 1')
plt.plot(t[13:23], f(t[13:23], *params2), 'b-', label='Fit 2')
plt.plot(t[22:], f(t[22:], *params3), 'k-', label='Fit 3')
plt.xlabel(r't / $ [s]$')
plt.ylabel(r'$\ln(\frac{p(t) - p_E}{p_0 - p_E})$')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend()
#plt.savefig('drehevak.pdf')
plt.show()