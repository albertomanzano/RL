import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr, erf

def delta(x,r,sigma,K,T):
    m = 1./(sigma*np.sqrt(2*T))
    c = m*(-np.log(K)+(r+sigma*sigma/2)*T)
    
    d1 = np.sqrt(2)*m*x+c
    return ndtr(d1)

def delta_fourier(x,r,sigma,K,T,a,b,N):
    m = 1./(sigma*np.sqrt(2*T))
    c = m*(-np.log(K)+(r+sigma*sigma/2)*T)
    shifted_a = a+c/m 
    shifted_b = b+c/m
    
    omega_0 = 2*np.pi/(b-a)
    
    delta = 1/(b-a)*(erf_int(shifted_b,m)-erf_int(shifted_a,m))*np.ones_like(x)
    for i in range(1,N):
        omega = omega_0*i
        
        sin_int = erf_sin_int(shifted_b,m,omega)-erf_sin_int(shifted_a,m,omega)
        cos_int = erf_cos_int(shifted_b,m,omega)-erf_cos_int(shifted_a,m,omega)
        print("Cos int: ",cos_int)

        sin_c = 2./(b-a)*(np.cos(omega*c/m)*sin_int - np.sin(omega*c/m)*cos_int)
        cos_c = 2./(b-a)*(np.cos(omega*c/m)*cos_int + np.sin(omega*c/m)*sin_int)

        delta = delta+sin_c*np.sin(omega*x)+cos_c*np.cos(omega*x)

    delta = 0.5*delta+0.5
    return delta
def erf_int(z,a,b = 1.):
    return z*erf(a*z)+1./(a*np.sqrt(np.pi))*np.exp(-a*a*z*z)

def erf_sin_int(z,a,b):
    term1 = -1./b*np.cos(b*z)*erf(a*z)
    term2 = 1/(2*b)*np.exp(-b*b/(4*a*a))
    term3 = erf(a*z-1j*b/(2*a))+erf(a*z+1j*b/(2*a))
    return term1+term2*np.real(term3)

def erf_cos_int(z,a,b):
    term1 = 1./b*np.sin(b*z)*erf(a*z)
    term2 = 1j/(2*b)*np.exp(-b*b/(4*a*a))
    term3 = erf(a*z-1j*b/(2*a))-erf(a*z+1j*b/(2*a))
    return term1+np.real(term2*term3)


# Market info
r = 0.01
sigma = 0.1
K = 1.
T = 0.1

# Domain definition
a = np.log(0.5)
b = np.log(1.5)
N = 15
n_points = 1000
x = np.linspace(a/2,b/2,n_points)

# Values for the delta
delta_exact = delta(x,r,sigma,K,T)
delta_approx = delta_fourier(x,r,sigma,K,T,a,b,N)

# Plot
plt.plot(x,delta_exact, label = "Exact")
plt.plot(x,delta_approx, label = "Approx")
plt.grid()
plt.legend()
plt.show()

