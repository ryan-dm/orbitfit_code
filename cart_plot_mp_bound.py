import numpy as np

#Ensure machines without displays can still render & save figures
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt

#Need coordinate conversions, cartesian integrator, keplerian solver
from cart_to_kepler import *
from RK4_energy import *
from koe_hyper_agnostic import koe

#For plotting
from matplotlib.ticker import MaxNLocator
from corner import corner

#For convergence checking
from grstat import gr
from chainplot import chainplot

#Useful to have for debugging
import pdb

#Date conversions
import jdcal as jdcal
import datetime as dtt

G = 4*np.pi**2 #In solar mass-AU-year coordinates. Important constant.

filename = 'data_pztel_reproc.csv' #path to astrometry file
save_file_base = '/big_scr7/dryan/pztelmc/pztel_cart_reproc1' #path to base filename of orbitfit npz saves
num_files = 7 #Number of orbitfit saves to import

#Read in data. NOTE: set usecols kwarg to the relevant columns for your file!
#Need epochs in MJD, xy and errors in arcseconds
#Due to old convention, data are interleaved so that even array indices are x,
#and odd array indices are y
epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(0,1), unpack=True)).T.ravel()
xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(7,8), unpack=True)).T.ravel()/1000
sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()/1000

#Load all npz saves. Thin by 75 (slightly larger than typical max autocorrelation)
#(Thinning can be adjusted as desired)
for i in range(num_files):
    with np.load(save_file_base+'_'+str(i)+'.npz') as d:
        if i==0: #Create variables if this is the first savefile
            chain = d['chain'][:,::75,:] #Shape is (nwalkers, nsteps, ndim)
            lnp = d['lnp'][:,::75]
        else: #Append to variables if they already exist
            chain = np.append(chain,d['chain'][:,::75,:],axis=1)
            lnp = np.append(lnp,d['lnp'][:,::75],axis=1)

#Get the output filename for the figure
tmp=save_file_base.split('/')
outname=tmp[-1]

#NOTE: Convergence checking only performed in the cart_plot_mp script
# This script plots only a subset of the values from cart_plot_mp, so
# you should run that script if you want Gelman-Rubin or chain visualizations

#Make copy of MCMC parameter chain
new_chain = chain.copy()
#Fill that copy with KEPLERIAN elements rather than CARTESIAN state vectors
for i in range(chain.shape[1]):
    for j in range(128):
        new_chain[j,i,:6] = cart_to_kepler(chain[j,i,:],epochs[0])

#For later plotting: save this variable in a convenient shape
#Important to save this BEFORE converting parallax->dist or radians->degrees
#So that it will work with plotting functions
koe_chain = new_chain.copy().reshape(-1,chain.shape[-1])

#cart_to_kepler returns angles in radians, so change to degrees
new_chain[:,:,2] = np.degrees(new_chain[:,:,2])%360.
new_chain[:,:,3] = np.degrees(new_chain[:,:,3])%360.
new_chain[:,:,4] = np.degrees(new_chain[:,:,4])%360.

#Invert parallax to get distance
new_chain[:,:,7] = 1./new_chain[:,:,7]

#Reshape new_chain for plotting covariances in corner()
samples = new_chain.reshape(-1,new_chain.shape[-1])
lnpflat = lnp.reshape(-1)


#Limit to bound orbits
w2 = np.where(samples[:,5]<1)[0] #Eccentricity <1
#NOTE: If there are INSUFFICIENT ALLOWED BOUND ORBITS, the computation will
# FAIL at this step. This can occur for data that is ONLY well-fit by hyperbolae.
# In that case, there's no point running this script, as there's no bound orbits
# available to view.
samples = samples[w2,:]
lnpflat = lnpflat[w2]
koe_chain = koe_chain[w2,:]

#Location of maximum likelihood
w = np.argmax(lnpflat)

#Chain corresponding to maximum likelihood
ml_chain = koe_chain[w,:]

#Labels of keplerian elements for corner plot
labels=["q[AU]",r'$\tau$',r'$\omega[^{\circ}]$',r'$\Omega[^{\circ}]$',r'$i[^{\circ}]$',r'$e$','M','dist']
#Generate corner plot (plot_datapoints=False speeds computation by preventing
# the drawing of several million unnessecary dots)
fig=corner(samples, labels=labels, show_titles=True,plot_datapoints=False)
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)

#Stick the orbit/astrometry plot in the top right
plt2=fig.add_subplot(3,3,3)

#### Generate Epochs for the lines ####
#Set up
MJD0 = 2400000.5   # zero point for JD
# generate 100 evenly spaced dates for plotting orbit curves
# padding extends to either side of observed epochs 
epo_pad = 365*7
epo = np.linspace(np.nanmin(epochs)-epo_pad,np.nanmax(epochs)+epo_pad,num=100)

#Duplicates dates for (x,y) plotting convention
epo = np.column_stack((epo,epo)).flatten()
#convert epochs to python datetime stucture (easier to understand than MJD)
ddt = np.array([])
for ep in epo:
    y,m,d,h = jdcal.jd2gcal(MJD0, ep)
    ddt = np.append(ddt,dtt.datetime(y,m,d))

# convert observed dates for plotting
epochdt = np.array([])
for ep in epochs:
    y,m,d,h = jdcal.jd2gcal(MJD0, ep)
    epochdt = np.append(epochdt,dtt.datetime(y,m,d))

#Plot a horizontal line at 0
plt.axhline(y=0,ls='--',color='k')

#Plot npts random orbits drawn from the MCMC samples
npts = 50
rindex = np.floor(np.random.random(npts) * np.size(lnpflat)).astype('int')
rindex = np.array([int(r) for r in rindex])
rindex = rindex.astype('int') #be VERY SURE the array indices are integers
for i in rindex:
    XY = koe(epo,*koe_chain[i,:]) #Generate XY coords from keplerian elements
    plt2.plot(ddt[0::2],XY[0::2],'r:',linewidth=1, alpha=0.05) #Plot x
    plt2.plot(ddt[1::2],XY[1::2],'b:',linewidth=1, alpha=0.05) #Ploy y


#Plot the most likely solution
XY=koe(epo,*ml_chain)
plt.plot(ddt[0::2],XY[0::2],'k',linewidth=1.5) #x
plt.plot(ddt[1::2],XY[1::2],'k',linewidth=1.5) #y

#Plot x astrometry
plt2.errorbar(epochdt[0::2],xy[0::2],yerr=sig[0::2],fmt='ro',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[0::2],xy[0::2],'ro',mew=1.5,label=r'X')
plt2.legend(numpoints=1,markerscale=2)

#Plot y astrometry
plt2.errorbar(epochdt[1::2],xy[1::2],yerr=sig[1::2],fmt='bo',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[1::2],xy[1::2],'bo',mew=1.5,label=r'Y')
plt2.legend(numpoints=1,markerscale=2)

### Set titles and limits ###
plt.xlim(dtt.datetime(2002, 1,1),dtt.datetime(2019, 1,1))
plt.ylabel('Offset [arcsec]')
plt.xlabel('Date [year]')

#Get the output filename for the figure
tmp=save_file_base.split('/')
outname=tmp[-1]

#Save the figure
plt.savefig(outname+'.png')
plt.clf()