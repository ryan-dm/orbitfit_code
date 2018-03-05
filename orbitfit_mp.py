#!/usr/bin/anaconda
import numpy as np
from emcee import PTSampler
import time
#from koe_mass import koe
from datetime import datetime,date
#import jdcal
import pdb
#from orbitclass import orbit
import matplotlib.pyplot as plt
import pickle as pickle 

#Can still use hyperbolic Kepler's eqn solver here, since behavior is identical for e<1.
from ema_hyper import ema

#Note! Priors, astrometry file, and walker initial positions are
#currently set up for PZ Tel. These should be changed for other systems.

savename='/big_scr7/dryan/pztelmc/pztel_mass_reproc1'

kgauss    =  np.double(0.017202098950) # Gauss's constant for orbital 
                                        # motion ... (kp)^2 = (2pi)^2 a^3



def koe(epochs,a,tau,argp,lan,inc,ecc,mass,para):

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    # Stellar properties for beta Pic

    mstar = mass
    
    parallax = para
    distance = 1./parallax
    ndate = len(epochs)

    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # a           semimajor axis [au]
    # tau         epoch of peri in units of the orbital period
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
    #
    # Derived quantities 
    # manom   --- mean anomaly
    # eccanom --- eccentric anomaly
    # truan   --- true anomaly
    # theta   --- longitude 
    # radius  --- star-planet separation

    n = kgauss*np.sqrt(mstar)*(a)**(-1.5)  # compute mean motion in rad/day

    # ---------------------------------------
    # Compute the anomalies (all in radians)
    #
    # manom = n * (epochs - tau) # mean anomaly 

    manom = n*epochs[0::2] - 2*np.pi*tau  # Mean anomaly w/tau in units of period

    eccanom = np.array([])
 
    for man in manom:
        #print man #DEBUGGING
        eccanom = np.append(eccanom, ema(man%(2*np.pi), ecc))

    # ---------------------------------------
    # compute the true  anomaly and the radius
    #
    # Elliptical orbit only

    truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    theta = truan + argp
    radius = a * (1.0 - ecc * np.cos(eccanom))

    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    # standard form
    #
    #xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    #yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    #
    # write in terms of \Omega +/- omega --- see Mathematica notebook trig-id.nb
        
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

    xp = radius*(c2i2*s1 - s2i2*s2)
    yp = radius*(c2i2*c1 + s2i2*c2)

    # Interleave x & y
    # put x data in odd elements and y data in even elements

    data = np.zeros(ndate)
    data[0::2]  = xp
    data[1::2]  = yp

    return data*parallax # results in seconds of arc



def logl(theta,eps,xy,sig):
    loga = theta[0]
    tau  = theta[1]
    argp = theta[2]
    lan  = theta[3]
    cinc = theta[4]
    ecc  = theta[5]
    mass = theta[6]
    para = theta[7]

    
    a = np.exp(loga)
    inc = np.arccos(cinc)
    
    XY=koe(eps,a,tau,argp,lan,inc,ecc,mass,para)
    
    lpx = -0.5*np.log(2*np.pi) * xy.size + \
                   np.sum( -np.log(sig)- 0.5*( (xy-XY)/sig)**2)
    
    return lpx

def logp(theta): #FOR 51 eri

    loga  = theta[0]
    tau   = theta[1]
    argp  = theta[2]
    lan   = theta[3]
    cinc  = theta[4]
    ecc   = theta[5]
    mass  = theta[6]
    para  = theta[7]

    # include priors here

    if (loga < np.log(5) or loga > np.log(5000)): 
        return -np.inf

    if (tau < -0.5 or tau > 0.5):
        return -np.inf
 
    if (argp < 0 or argp > 2*np.pi):
        return -np.inf

    if (lan < 0 or lan > np.pi*2):
        return -np.inf

    if (cinc < -1 or cinc > 1): 
        return -np.inf

    #if (ecc < 0 or ecc >= 0.95724):
    #if (ecc <0.4 or ecc >= 1.0):
    if (ecc<0. or ecc>=1.):
        return -np.inf
    
    if mass<0.5 or mass>2.5:
        return -np.inf

    if para<10.e-3 or para>40e-3:
        return -np.inf

    parachi2= -((para - 19.42e-3)/0.98e-3)**2
    masschi2= -((mass - 1.2)/0.1)**2 # TNH 2011, see Ginski 14 S2.2

    # otherwise ... 
    return parachi2+masschi2
    #return np.log(-2.1826563*ecc+2.0893331)
print 'likelihoods & priors built!'
######################################################## 

ntemps   = 20 #Number of temperatures for the parallel tempering
nwalkers = 128 #Number of walkers
ndim     = 8 #Number of dimensions in the orbit fit

#filename='51eri.csv' #Beta Pic
#filename='fomalhaut_new.csv' #Fomalhaut
#filename = 'hr8799b_mjd.csv'
filename = 'data_pztel_reproc.csv'
#filename = 'astrometry_csv2.csv'
eps=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(0,1), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(7,8), unpack=True)).T.ravel()/1000
sigs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()/1000
eps=eps[xy == xy]
sigs=sigs[xy == xy]
xy=xy[xy == xy]
#pdb.set_trace()
'''
#FOR FOMALHAUT
w0 = np.random.uniform( 4.0,7.0,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.25,0.25,    size=(ntemps,nwalkers))
w2 = np.random.uniform(-np.pi/4.,3.*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( np.pi/3,np.pi,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.3,0.3,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.785,0.825,   size=(ntemps,nwalkers))
'''
'''
#FOR BETA PIC
w0 = np.random.uniform( 2.0,2.5,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.25,0.25,    size=(ntemps,nwalkers))
w2 = np.random.uniform(np.pi/4.,3*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( 0.8,1.2,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.05,0.05,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.05,0.2,   size=(ntemps,nwalkers))
w6 = np.random.uniform(1.7,1.8, size=(ntemps,nwalkers))
'''
'''
#for 51 eri
w0 = np.random.uniform( 2.3,3,    size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.1,0.1,    size=(ntemps,nwalkers))
w2 = np.random.uniform(np.pi/4.,3*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform( 0.8,1.2,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.1,0.1,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.1,0.2,   size=(ntemps,nwalkers))
'''
'''
#for hr8799b
w0 = np.random.uniform( np.log(12),np.log(20),   size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.1,0.1,    size=(ntemps,nwalkers))
w2 = np.random.uniform(-np.pi/4.,np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform(-np.pi/4.,np.pi/4.,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.1,0.1,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.1,0.2,   size=(ntemps,nwalkers))
'''

#for PZ Tel
w0 = np.random.uniform( np.log(40),np.log(60),   size=(ntemps,nwalkers))
w1 = np.random.uniform( -0.1,0.1,    size=(ntemps,nwalkers))
w2 = np.random.uniform(np.pi/4.,5*np.pi/4.,  size=(ntemps,nwalkers))
w3 = np.random.uniform(np.pi/4.,5*np.pi/4.,    size=(ntemps,nwalkers))
w4 = np.random.uniform(-0.3,0.3,  size=(ntemps,nwalkers))
w5 = np.random.uniform( 0.6,0.8,   size=(ntemps,nwalkers))
w6 = np.random.uniform(1.1,1.2,size=(ntemps,nwalkers))
w7 = np.random.uniform(18.42e-3,20.42e-3,size=(ntemps,nwalkers))

p0 = np.dstack((w0,w1,w2,w3,w4,w5,w6,w7))

#with np.load('/big_scr7/dryan/pztelmc/pztel_mass_300k/pztel_mass_300k_9.npz') as d:
#    p0[:,:,:7] = d['plast']

#Numnber of iterations (after the burn-in)
niter = 3000000
num_files=100 #I trust YOU, user, to make sure niter / num_files is remainderless

nperfile=niter/num_files
thin=1
#Burn-in for 1/5 of the number of iterations
nburn = 50000
print 'Made it this far, waiting 5 secs'
if __name__=='__main__':
    time.clock()
    time.sleep(1)
    startTime = datetime.now()
    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp ,loglargs=[eps,xy,sigs],threads=16)
    sampler.run_mcmc(p0,nburn,thin=thin)
    for index in range(num_files):
        if index == 0:
            p = sampler.chain[:,:,-1,:]
            np.savez_compressed(savename+'_burn_'+str(index),af=sampler.acceptance_fraction[0],
                chain=sampler.chain[0],lnp=sampler.lnprobability[0],plast=p)
            print 'Burn in complete'
            #burnchain = sampler.chain
            #pdb.set_trace()
        else:
            p = sampler.chain[:,:,-1,:]
        sampler.reset()
        #print 'Burn in complete'
        sampler.run_mcmc(p,nperfile,thin=thin)
        #print 'orbit fitting complete'

        if index == num_files-1:
            SMA = np.exp(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,0]),[16,50,84]))
            LAN = np.degrees(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,3]),[16,50,84]))
            INC = np.degrees(np.arccos(np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,4]),[16,50,84])))
            ECC = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,5]),[16,50,84])
            mstar = np.percentile(np.ndarray.flatten(sampler.flatchain[0,:,6]),[16,50,84])
            PER = 2*np.pi/(kgauss*np.sqrt(mstar)*(SMA)**(-1.5) ) /365.25
            print 'SMA = ',SMA,'; AU'
            print 'LAN = ',LAN,' deg'
            print 'inc = ', INC,' deg'
            print 'ecc = ',ECC
            print 'Per = ',PER,' yr'


        np.savez_compressed(savename+'_'+str(index),af=sampler.acceptance_fraction[0],
                            chain=sampler.chain[0],lnp=sampler.lnprobability[0],plast=sampler.chain[:,:,-1,:])

    
    print 'Total Execution Time:', (datetime.now()-startTime)
    #pdb.set_trace()
