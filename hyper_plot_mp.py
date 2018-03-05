import numpy as np

#Ensure machines without displays can still render & save figures
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt

#For plotting
from matplotlib.ticker import MaxNLocator
from corner import corner

#Need Kepler's eqn solver
from ema_hyper import ema

#For convergence checking
from grstat import gr
from chainplot import chainplot

#Useful to have for debugging
import pdb

#Date conversions
import jdcal as jdcal
import datetime as dtt

#Convert errors from x/y space to sep/PA space
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

#Convert x/y positions to sep/PA
def xy2seppa(xy):
    xs = xy[0::2]
    ys = xy[1::2]
    sep = np.sqrt(xs**2 + ys**2)
    pa = np.arctan2(xs,ys)%(2*np.pi)
    return sep,pa

#Convert sep/PA to x/y positions
def seppa2xy(sep,pa):
    xs = sep*np.sin(pa)
    ys = sep*np.cos(pa)
    xy = np.zeros(2*len(xs))
    xy[0::2]=xs
    xy[1::2]=ys
    return xy

#Hyperbolic orbit fit currently generates tau in a weird way, since it was
#extended from the bound/elliptical orbitfit where tau is period-folded.
#Hyperbolic orbitfit keeps tau in those dimensionless units, but there is
#no longer a period to fold over. This function assumes dimensionless tau.
#For a function that assumes tau is in MJD, see file koe_hyper_agnostic.py
def koe(epochs,q,tau,argp,lan,inc,ecc,mass,para):

    # Epochs in MJD
    # 
    # Epochss are doubled due to x,y interleaving w/in a single array

    kgauss    =  np.double(0.017202098950)
    mstar = mass
    parallax = para
    distance = 1./parallax
    ndate = len(epochs)
    if ecc!=1.0:
        a = np.abs(q/(1-ecc))
    # Keplerian Elements 
    #
    # epochs      dates in modified JD [day]
    # q           periastron dist [au]
    # tau         epoch of peri in folded units
    # argp        argument of peri [radians]
    # lan         longitude of ascending node [radians]
    # inc         inclination [radians]
    # ecc         eccentricity 
    # mass
    # para
    #
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

    manom = n*epochs[0::2] - 2*np.pi*tau  # Mean anomaly w/tau in units of period

    eccanom = np.array([])
 
    for man in manom:
        #print man #DEBUGGING
        eccanom = np.append(eccanom, ema(man, ecc))

    if ecc>1:
        truan = np.arccos((ecc-np.cosh(eccanom))/(ecc*np.cosh(eccanom)-1.))*np.sign(eccanom)
    elif ecc<1:
        truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom) )
    else:
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

    return data*parallax


#Astrometry
#filename = 'astrometry_csv2.csv'
filename = 'data_pztel_reproc.csv'
#filename = '51eri.csv'
#filename = 'fomalhaut_new.csv'

if filename =='data_pztel_reproc.csv':
    epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(0,1), unpack=True)).T.ravel()
    xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(7,8), unpack=True)).T.ravel()/1000
    sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()/1000

else:
    epochs=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(9,10), unpack=True)).T.ravel()
    xy=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(11,13), unpack=True)).T.ravel()/1000
    sig=(np.genfromtxt(filename, skip_header=1,delimiter=',', usecols=(12,14), unpack=True)).T.ravel()/1000

#Get the output filename for the figure
tmp=save_file_base.split('/')
outname=tmp[-1]

##Convergence checking::
#Compute gelman-rubin statistic for each fit parameter & write to file
fff = open(outname+'_grvals.txt','wb')
for i in range(chain.shape[2]):
    fff.write('MC parameter '+str(i)+': GR '+str(gr(chain[:,:,i]))+'\n')
fff.close()
#Plot chains (with extra thinning for legiiblity) to check convergence
chainplot(chain,outname,thin=100)

#save_file_base = '/big_scr7/dryan/betapicmc/bp_mass_300k'
save_file_base = '/big_scr7/dryan/pztelmc/pztel_hyper_reproc1'
num_files = 36
for i in range(num_files):
    with np.load(save_file_base+'_'+str(i)+'.npz') as d:
        if i==0:
            chain = d['chain']
            lnp = d['lnp']
        else:
            chain = np.append(chain,d['chain'],axis=1)
            lnp = np.append(lnp,d['lnp'],axis=1)

chain = chain[:,::75,:]
lnp = lnp[:,::75]

#chain[:,:,1] = chain[:,:,1]%1.
chain[:,:,0] = np.exp(chain[:,:,0])
chain[:,:,4] = np.arccos(chain[:,:,4])
koe_chain = chain.copy().reshape(-1,chain.shape[-1])

#if chain.shape[-1]==7:
#    P = np.sqrt(chain[:,:,0]**3./chain[:,:,6])*365.2425
#else:
#    P = np.sqrt(chain[:,:,0]**3./1.13)*365.2425
#chain[:,:,1]=((chain[:,:,1]*P)+50000-49718)%P+49718


chain[:,:,2] = np.degrees(chain[:,:,2])%360.
chain[:,:,3] = np.degrees(chain[:,:,3])%360.
chain[:,:,4] = np.degrees(chain[:,:,4])%360.
chain[:,:,7] = 1./chain[:,:,7]


samples = chain.reshape(-1,chain.shape[-1])
lnpflat = lnp.reshape(-1)

w = np.argmax(lnpflat)

ml_chain = koe_chain[w,:]


labels=["q[AU]",r'$\tau$',r'$\omega[^{\circ}]$',r'$\Omega[^{\circ}]$',r'$i[^{\circ}]$',r'$e$','M','dist']

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
rindex = np.floor(np.random.random(npts) * np.size(lnp)).astype('int')
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



#Save the figure
plt.savefig(outname+'.png')
plt.clf()
#plt.show()

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
