import numpy as np
from emcee import PTSampler

#The following MAY be necessary to deal with an update that changed
#how python multiprocessing default parameters work. Without this line,
#exponentially too many threads are spawned.
import mkl
mkl.set_num_threads(1)

import time
from datetime import datetime,date
#import jdcal
import pdb
#from orbitclass import orbit
import matplotlib.pyplot as plt
import pickle as pickle 
from scipy.integrate import odeint
from RK4_energy import *
G = 4.*np.pi**2.

#Note! Priors, astrometry file, and walker initial positions are
#currently set up for PZ Tel. These should be changed for other systems.
kgauss    =  np.double(0.017202098950) # Gauss's constant for orbital 
                                        # motion ... (kp)^2 = (2pi)^2 a^3

savename = '/big_scr7/dryan/pztelmc/pztel_cart_reproc1'
#savename = 'cart_test2'
filename = 'data_pztel_reproc.csv'

eps=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=0, unpack=True)).T.ravel()
x=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=7, unpack=True)).T.ravel()/1000
y=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=8, unpack=True)).T.ravel()/1000
sigx=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=9, unpack=True)).T.ravel()/1000
sigy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=10, unpack=True)).T.ravel()/1000
eps=eps[x == x]/365.2425
sigx=sigx[x==x]
sigy=sigy[x==x]
y=y[x==x]
x=x[x==x]

ntemps   = 20 #Number of temperatures for the parallel tempering
nwalkers = 128 #Number of walkers
ndim     = 8 #Number of dimensions in the orbit fit


def sp_e(x0,G,M):
    return sp_kin_e(x0)+sp_pot_e(x0,G,M)

def logl(theta,eps,x,y,sigx,sigy):
    x0 = theta[:6]
    M = theta[6]
    para=theta[7]
    dist=1./para

    xd = x*dist
    yd = y*dist
    sigxd = sigx*dist
    sigyd = sigy*dist

    X,Y = interp_xy(deriv_function,eps,x0,dt,G,M)
    #out_array = odeint(deriv_function,x0,eps,args=(G,M),Dfun=fun_jacob,col_deriv=1)/distance
    #X = out_array[:,0]
    #Y = out_array[:,1]

    lp = -0.5*np.log(2*np.pi)*(xd.size+yd.size) + \
        np.sum(-np.log(sigxd)-np.log(sigyd)-0.5*((xd-X)/sigxd)**2-0.5*((yd-Y)/sigyd)**2)

    return lp

def logp(theta,eps,x,y,sigx,sigy):
    x0 = theta[0]
    y0 = theta[1]
    z0 = theta[2]
    vx0 = theta[3]
    vy0 = theta[4]
    vz0 = theta[5]
    M = theta[6]
    para = theta[7]

    if x0<0 or x0>300:
        return -np.inf

    if y0<0 or y0>300:
        return -np.inf

    if z0<-1000 or z0>1000:
        return -np.inf

    if vx0<0 or vx0>10:
        return -np.inf

    if vy0<0 or vy0>10:
        return -np.inf

    if vz0<-10 or vz0>10:
        return -np.inf

    if M<0.5 or M>2.5:
        return -np.inf

    if para<10e-3 or para>40e-3:
        return -np.inf

    parachi2= -((para - 19.42e-3)/0.98e-3)**2
    masschi2= -((M - 1.2)/0.1)**2 # TNH 2011, see Ginski 14 S2.2

    return parachi2 + masschi2

#Variables for computing initial positions of walkers
X0 = x[0]*distance
Y0 = y[0]*distance
PA = np.arctan2(X0,Y0)
sX0 = sigx[0]*distance
sY0 = sigy[0]*distance
Z0 = 0.0
sZ0 = np.mean([sX0,sY0])
PE = sp_pot_e([X0,Y0,Z0,0,0,0],G,1.2)
speed_square = np.abs(PE)
dx0 = np.sqrt(speed_square/2.)
dy0 = np.sqrt(speed_square/2.)
dz0 = 0

#Create initial positions of walkers
w0 = np.random.uniform( X0-sX0,X0+sX0,    size=(ntemps,nwalkers))
w1 = np.random.uniform( Y0-sY0,Y0+sY0,    size=(ntemps,nwalkers))
w2 = np.random.uniform( Z0-sZ0,Z0+sZ0,    size=(ntemps,nwalkers))
w3 = np.random.uniform(dx0*0.99,dx0*1.01,    size=(ntemps,nwalkers))
w4 = np.random.uniform(dy0*0.99,dy0*1.01,  size=(ntemps,nwalkers))
w5 = np.random.uniform(-0.01,0.01,size=(ntemps,nwalkers))
w6 = np.random.uniform(1.2,1.3, size=(ntemps,nwalkers))
w7 = np.random.uniform(18.42e-3,20.42e-3,size=(ntemps,nwalkers))

#Stack them together in an 8-parameter set
p0 = np.dstack((w0,w1,w2,w3,w4,w5,w6,w7))

#Numnber of iterations (after the burn-in)
niter = 1500000
num_files=50 #I trust YOU, user, to make sure niter / num_files is remainderless

nperfile=niter/num_files
thin=1 #Thinning; if 1, accepts every sample
nburn = 50000 #Burn-in

if __name__=='__main__':
    print 'Beginning main function'
    time.clock()
    time.sleep(1)
    startTime = datetime.now()
    sampler=PTSampler(ntemps, nwalkers, ndim, logl, logp ,loglargs=[eps,x,y,sigx,sigy],logpargs=[eps,x,y,sigx,sigy],threads=32)
    sampler.run_mcmc(p0,nburn,thin=thin)
    for index in range(num_files):
        if index == 0:
            p = sampler.chain[:,:,-1,:]
            np.savez_compressed(savename+'_burn_'+str(index),af=sampler.acceptance_fraction[0],
                chain=sampler.chain[0],lnp=sampler.lnprobability[0],p=sampler.chain[:,:,-1,:])
            print 'Burn in complete'
            #burnchain = sampler.chain
            #pdb.set_trace()
        else:
            p = sampler.chain[:,:,-1,:]
        sampler.reset()
        #print 'Burn in complete'
        sampler.run_mcmc(p,nperfile,thin=thin)
        #print 'orbit fitting complete'

        np.savez_compressed(savename+'_'+str(index),af=sampler.acceptance_fraction[0],
                            chain=sampler.chain[0],lnp=sampler.lnprobability[0],p=sampler.chain[:,:,-1,:])

    
    print 'Total Execution Time:', (datetime.now()-startTime)
    #pdb.set_trace()
