import numpy as np
from ema_hyper import ema

#NOTE: Functions with only BOUND orbits may use a convention where tau is
#folded over the orbital period & constrained b/w [0,1] or [-0.5,0.5]. THIS
#function requires that tau be provided in MJD, as it is "agnostic" to
#any possible wrappings the user may wish to do, and cares only about dates.
def koe(epochs,q,tau,argp,lan,inc,ecc,mass,para):

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    kgauss    =  np.double(0.017202098950)
    mstar = mass
    parallax = para
    distance = 1./parallax
    ndate = len(epochs)
    if ecc!=1.0: #Non-parabolic case: want a semimajor axis
        a = np.abs(q/(1-ecc))
    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # q           periastron distance [au]
    # tau         epoch of peri [MJD]
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
    # mass        System mass [m_sun]
    # para        parallax [arcsec]

    # Derived quantities 
    # manom   --- mean anomaly
    # eccanom --- eccentric anomaly
    # truan   --- true anomaly
    # theta   --- longitude 
    # radius  --- star-planet separation

    if ecc!=1.0:
        n = kgauss*np.sqrt(mstar)*(a)**(-1.5)  # compute mean motion in rad/day
    else:
        n = kgauss*np.sqrt(mstar)*(2*q)**(-1.5)
    # ---------------------------------------
    # Compute the anomalies (all in radians)
    #
    # manom = n * (epochs - tau) # mean anomaly 

    manom = n*(epochs[0::2]-tau)  # Mean anomaly w/tau in units of period

    eccanom = np.array([])
 
    for man in manom:
        #ema function treats e<1, e=1, e>1 all correctly
        eccanom = np.append(eccanom, ema(man, ecc))

    if ecc>1: #Hyperbola
        truan = np.arccos((ecc-np.cosh(eccanom))/(ecc*np.cosh(eccanom)-1.))*np.sign(eccanom)
    elif ecc<1: #Ellipse
        truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    else: #Parabola
        truan = np.arctan(2*eccanom)
    # ---------------------------------------
    # compute the true  anomaly and the radius
    #
    # Elliptical orbit only

    theta = truan + argp
    p = q*(1+ecc)
    radius = p/(1+ecc*np.cos(truan))
    
    
    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    # standard form
    #
    #xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    #yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    #
        
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2

    arg0 = truan + lan
    arg1 = truan + argp + lan
    arg2 = truan + argp - lan
    arg3 = truan - lan

    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)

    sa0 = np.sin(arg0)
    sa3 = np.sin(arg3)
    
    # updated sign convention for Green Eq. 19.4-19.7
    # Compute actual x & y
    xp = radius*(c2i2*s1 - s2i2*s2)
    yp = radius*(c2i2*c1 + s2i2*c2)

    # Interleave x & y
    # put x data in odd elements and y data in even elements

    data = np.zeros(ndate)
    data[0::2]  = xp
    data[1::2]  = yp

    return data*parallax #Return interleaved data*parallax to get desired units