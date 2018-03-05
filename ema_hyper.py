import numpy as np
import pdb

#Set of functions, adapted from Rob de Rosa's implementation of Mikkola et al 87
#For full explanation of operations involved, see that paper
#Solves Kepler's eqn to high accuracy with a power series expansion, instead of a loop
#Works for all eccentricities
def ema(M,e):
    if e==0: return M%(2*np.pi)
    elif M==0: return M
    elif e<1:
        if M%(2*np.pi)==0: return 0.
        else: return ellipse(M%(2*np.pi),e)
    elif e==1: return parabola(M)
    else: return hyperbola(M,e)

def ellipse(Manom,e): #Elliptical case
    flag = False
    if Manom > np.pi:
        flag = True
        Manom = 2.*np.pi - Manom
    alpha = (1. - e)/(4.*e + 0.5)
    beta = 0.5*Manom / (4. * e + 0.5)
    aux = np.sqrt(beta**2. + alpha**3.)
    z = beta + aux
    z = z**(1./3.)

    s0 = z - (alpha/z)
    
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + e)
    e0 = Manom + (e * (3.0*s1 - 4.0*(s1**3.0)))
    
    se0 = np.sin(e0)
    ce0 = np.cos(e0)
    
    f = e0-e*se0-Manom
    f1 = 1.0-e*ce0
    f2 = e*se0
    f3 = e*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.1666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.1666666666667*f3*u3*u3+0.04166666666667*f4*(u3**3.0))
    
    Eanom = (e0 + u4)
    if flag == True:
        Manom = 2.*np.pi - Manom
        Eanom = 2.*np.pi - Eanom
    return Eanom

def hyperbola(Manom,e): #Hyperbolic case
    alpha = (e-1.)/(4.*e+0.5)
    beta = 0.5*Manom / (4.*e + 0.5)
    aux = np.sqrt(beta**2.+alpha**3.)
    if beta<0: z = beta - aux
    else: z = beta + aux
    if z<0: z = -((-z)**(1./3.))
    else: z = z**(1./3.)
    s0 = z - (alpha/z)
    s1 = s0 + (0.071*(s0**5.)) / ((1.0 + 0.45*s0**2.)*(1.0 + 4.0*s0**2.)*e)
    f0 = 3.0*np.log(s1+np.sqrt(1.0+s1**2.))

    sf0 = np.sinh(f0)
    cf0 = np.cosh(f0)
    f = e*sf0-f0-Manom
    f1 = e*sf0-1.0
    f2 = e*sf0
    f3 = e*cf0
    f4 = e*sf0
    f5 = e*cf0
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.1666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.1666666666667*f3*u3*u3+0.04166666666667*f4*(u3**3.0))
    u5 = -f/(f1+0.5*f2*u4+0.1666666666667*f3*u4*u4+0.04166666666667*f4*(u4**3.0)+0.008333333333333*f5*(u4**4.0))
    Fanom = f0+u5
    return Fanom

def parabola(M): #Parabolic case (not fn of e, since e==1 here)
    #Much easier computationally, but unlikely for MCMC to generate e EXACTLY 1
    paren = (np.sqrt(9.0*(M**2.0)+4.0) + 3.0*M)/2.0
    D = paren**(1./3.) - paren**(-1./3.)
    return D

vema = np.vectorize(ema)
vhyp = np.vectorize(hyperbola)

#For testing
if __name__=='__main__':
    uno = np.ones(201)
    theta_arr = np.linspace(-np.pi,np.pi,201)
    newe = np.linspace(1.,3.,201)
    thetatest = np.outer(uno,theta_arr)
    newetest = np.outer(newe,uno)
    theta_asymp = np.pi-np.arctan(np.sqrt(newetest**2 - 1.))
    for i in range(201):
        for j in range(201):
            if thetatest[i,j]>theta_asymp[i,j]:
                thetatest[i,j]=np.inf
    Ftest = np.arccosh((newetest+np.cos(thetatest))/(1.+newetest*np.cos(thetatest)))
    Ftest *= np.sign(thetatest)
    Mhtest = newetest*np.sinh(Ftest)-Ftest
    Frec = vhyp(Mhtest,newetest)
    Ferr = Frec-Ftest
    Frerr = Ferr/Ftest
    pdb.set_trace()