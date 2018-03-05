import numpy as np

#Compute Gelman-Rubin statistic according to GR 1992
def gr(chains): #Input: 2D array of MCMC samples in a SINGLE parameter
    shp = chains.shape
    m = shp[0] #nwalkers
    n = shp[1] #nsteps
    sjsqr = np.zeros(m) #Array of zeros to fill with variance within walker
    for j in range(m): #For each walker, compute sample variance within self
        sjsqr[j]=1./(n-1.)*np.sum((chains[j,:]-np.mean(chains[j,:]))**2)
    thetaj = np.mean(chains,axis=1) #Mean values of individual walkers
    meanmean = np.mean(chains) #Mean value of all samples from all walkers
    W = np.mean(sjsqr) #Average of all intra-walker variances
    B = n/(m-1.)*np.sum((thetaj-meanmean)**2) #Variance of walker means from global mean
    var = (1.-(1./n))*W + B/n
    return np.sqrt(var/W) #GR statistic