#############################################
import pickle as pickle
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import pylab as P
from matplotlib.ticker import MaxNLocator
from corner import corner
import pdb
#from koe_mass_pztel import koe
#from koe import koe
# from constants import constants
from etest import ema
import jdcal as jdcal
import datetime as dt
#############################################
def koe(epochs,a,tau,argp,lan,inc,ecc,mass,para):

    # Epochs in MJD
    # 
    # date twice but x & y are computed.  
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #
    # Stellar properties for beta Pic
    kgauss    =  np.double(0.017202098950)
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

def sigs_xy2sigs_pa(xy,sigs):
    xs = xy[0::2]
    ys = xy[1::2]
    xsig = sigs[0::2]
    ysig = sigs[1::2]
    sep,pa = xy2seppa(xy)
    sep_dx = xs/sep
    sep_dy = ys/sep
    pa_dx = 1./(1.+(xs/ys)**2)/ys
    pa_dy = -xs/(ys**2)/(1.+(xs/ys)**2)
    sep_sig = np.sqrt((sep_dx*xsig)**2 + (sep_dy*ysig)**2)
    pa_sig = np.sqrt((pa_dx*xsig)**2 + (pa_dy*ysig)**2)
    return sep_sig,pa_sig

def xy2seppa(xy):
    xs = xy[0::2]
    ys = xy[1::2]
    sep = np.sqrt(xs**2 + ys**2)
    pa = np.arctan2(xs,ys)%(2*np.pi)
    return sep,pa

def seppa2xy(sep,pa):
    xs = sep*np.sin(pa)
    ys = sep*np.cos(pa)
    xy = np.zeros(2*len(xs))
    xy[0::2]=xs
    xy[1::2]=ys
    return xy

#Astrometry
#filename = 'astrometry_csv2.csv'
filename = 'data_pztel_reproc.csv'
#filename = '51eri.csv'
#filename = 'fomalhaut_new.csv'

if filename =='data_pztel_reproc.csv' or filename=='data_pztel_nonici.csv':
    epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(0,1), unpack=True)).T.ravel()
    xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(7,8), unpack=True)).T.ravel()/1000
    sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()/1000

else:
    epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
    xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
    sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000

#save_file_base = '/big_scr7/dryan/betapicmc/bp_mass_300k'
#save_file_base = '/big_scr7/dryan/erimc/eri_mass_test/eri_mass_test'
#save_file_base = '/big_scr7/dryan/fmhmc/fmh_mass_300k'
save_file_base = '/big_scr7/dryan/pztelmc/pztel_mass_reproc1'
num_files = 34
for i in range(num_files):
    with np.load(save_file_base+'_'+str(i)+'.npz') as d:
        if i==0:
            chain = d['chain'][:,::75,:]
            lnp = d['lnp'][:,::75]
        else:
            chain = np.append(chain,d['chain'][:,::75,:],axis=1)
            lnp = np.append(lnp,d['lnp'][:,::75],axis=1)

chain = chain[:,::75,:]
lnp = lnp[:,::75]
#pdb.set_trace()
chain[:,:,1] = chain[:,:,1]%1.
chain[:,:,0] = np.exp(chain[:,:,0])
chain[:,:,4] = np.arccos(chain[:,:,4])
koe_chain = chain.copy().reshape(-1,chain.shape[-1])

#if chain.shape[-1]==7:
P = np.sqrt(chain[:,:,0]**3./chain[:,:,6])*365.2425
#else:
#    P = np.sqrt(chain[:,:,0]**3./1.13)*365.2425
chain[:,:,1]=((chain[:,:,1]*P)+50000-49718)%P+49718


chain[:,:,2] = np.degrees(chain[:,:,2])%360.
chain[:,:,3] = np.degrees(chain[:,:,3])%360.
chain[:,:,4] = np.degrees(chain[:,:,4])%360.
chain[:,:,7] = 1./chain[:,:,7]

samples = chain.reshape(-1,chain.shape[-1])
lnpflat = lnp.reshape(-1)

w = np.argmax(lnpflat)

ml_chain = koe_chain[w,:]


labels=["a[AU]",r'$\tau$',r'$\omega[^{\circ}]$',r'$\Omega[^{\circ}]$',r'$i[^{\circ}]$',r'$e$','M','dist']

fig=corner(samples, labels=labels, show_titles=True,plot_datapoints=False)

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)

#Stick the plot in the top right
plt2=fig.add_subplot(3,3,3)

#### Generate Epochs for the lines ####
#Set up
MJD0 = 2400000.5   # zero point for JD
DT = 50000
#MJD0 = MJD0+DT #Comment out for non-sub50k data
# generate range of dates 
epo_pad = 365*7
epo = np.linspace(np.nanmin(epochs)-epo_pad,np.nanmax(epochs)+epo_pad,num=100)

epo = np.column_stack((epo,epo)).flatten()
#convert epochs to datetime stucture
ddt = np.array([])
for ep in epo:
    y,m,d,h = jdcal.jd2gcal(MJD0, ep)
    ddt = np.append(ddt,dt.datetime(y,m,d))
# convert observed dates for plotting
epochdt = np.array([])

##### Convert the Data epochs #### 

for ep in epochs:
    y,m,d,h = jdcal.jd2gcal(MJD0, ep)
    epochdt = np.append(epochdt,dt.datetime(y,m,d))


#Plot a horizontal line at 0
plt.axhline(y=0,ls='--',color='k')

#Get the orbit and plot it for npts
npts = 50
rindex = np.floor(np.random.random(npts) * np.size(lnp))
rindex = rindex.astype('int')
for i in rindex:
        XY = koe(epo,*koe_chain[i,:])
        plt2.plot(ddt[0::2],XY[0::2],'r:',linewidth=1, alpha=0.05)
        plt2.plot(ddt[1::2],XY[1::2],'b:',linewidth=1, alpha=0.05)


#Plot the most likely solution
XY=koe(epo,*ml_chain)

plt.plot(ddt[0::2],XY[0::2],'k',linewidth=1.5)
plt.plot(ddt[1::2],XY[1::2],'k',linewidth=1.5)

#### Plot the position of the planet ####

plt2.errorbar(epochdt[0::2],xy[0::2],yerr=sig[0::2],fmt='ro',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[0::2],xy[0::2],'ro',mew=1.5,label=r'X')
plt2.legend(numpoints=1,markerscale=2)

plt2.errorbar(epochdt[1::2],xy[1::2],yerr=sig[1::2],fmt='bo',ecolor='k', elinewidth=2, capthick=2, ms=2)
plt2.plot(epochdt[1::2],xy[1::2],'bo',mew=1.5,label=r'Y')
plt2.legend(numpoints=1,markerscale=2)


### Set titles and limits ###
plt.xlim(dt.datetime(2002, 1,1),dt.datetime(2019, 1,1))
plt.ylabel('Offset [arcsec]')
plt.xlabel('Date [year]')

#Get the output filename for the figure
tmp=save_file_base.split('/')
outname=tmp[-1]

#Save the figure
plt.savefig(outname+'.png')
plt.clf()
XY = koe(epochs,*ml_chain)
chi2 = ((xy-XY)/sig)**2.0
#pdb.set_trace()
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.errorbar(epochdt[0::2],xy[0::2]-XY[0::2],yerr=sig[0::2],fmt='ro',ecolor='k',elinewidth=2,capthick=2,ms=2)
plt.plot(epochdt[0::2],xy[0::2]-XY[0::2],'ro',mew=1.5,label=r'X')
plt.legend(numpoints=1,markerscale=2)
plt.xlim(dt.datetime(2006,1,1),dt.datetime(2019,1,1))
plt.ylabel('x resid [arcsec]')
plt.axhline(y=0,ls='--',color='k')
plt.title('X/Y Residuals - Normal')

ax2 = fig.add_subplot(212)
plt.errorbar(epochdt[1::2],xy[1::2]-XY[1::2],yerr=sig[1::2],fmt='bo',ecolor='k',elinewidth=2,capthick=2,ms=2)
plt.plot(epochdt[1::2],xy[1::2]-XY[1::2],'bo',mew=1.5,label=r'Y')
plt.legend(numpoints=1,markerscale=2)
plt.xlim(dt.datetime(2006,1,1),dt.datetime(2019,1,1))
plt.ylabel('y resid [arcsec]')
plt.xlabel('Date [year]')
plt.axhline(y=0,ls='--',color='k')
plt.savefig(outname+'resid.png')
plt.clf()

sep,pa = xy2seppa(xy)
SEP,PA = xy2seppa(XY)
pa = np.degrees(pa)
PA = np.degrees(PA)
sigs,sigp = sigs_xy2sigs_pa(xy,sig)
sigp = np.degrees(sigp)
resid_sep = sep-SEP
resid_pa = pa-PA

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.errorbar(epochdt[0::2],resid_sep,yerr=sigs,fmt='ro',ecolor='k',elinewidth=2,capthick=2,ms=2)
plt.plot(epochdt[0::2],resid_sep,'ro',mew=1.5,label=r'Sep')
plt.legend(numpoints=1,markerscale=1)
plt.xlim(dt.datetime(2006,1,1),dt.datetime(2019,1,1))
plt.ylabel('sep resid [arcsec]')
plt.axhline(y=0,ls='--',color='k')
plt.title('Sep/PA Residuals - Normal')


ax2 = fig.add_subplot(212)
plt.errorbar(epochdt[1::2],resid_pa,yerr=sigp,fmt='bo',ecolor='k',elinewidth=2,capthick=2,ms=2)
plt.plot(epochdt[1::2],resid_pa,'bo',mew=1.5,label=r'PA')
plt.legend(numpoints=1,markerscale=1)
plt.xlim(dt.datetime(2006,1,1),dt.datetime(2019,1,1))
plt.ylabel('PA resid [degrees]')
plt.xlabel('Date [year]')
plt.axhline(y=0,ls='--',color='k')
plt.savefig(outname+'resid_seppa.png')
plt.clf()


#plt.show()
