import numpy as np
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt
#Above syntax ensures plots generate correctly even if machine has no monitor

#Inputs: (nwalkers,nsteps,ndim) chain; save file base name; extra thinning of provided chains
#Outputs: Saves plots of walkers wandering, returns nothing
def chainplot(chains,outname,thin=1):
    shp = chains.shape
    for j in range(shp[2]): #Generate 1 plot per MC Parameter
        for i in range(shp[0]): #Plot each walker's trace
            plt.plot(chains[i,::thin,j],color=(0.,0.,0.,0.05)) #At very low alpha, ensuring transparency
        plt.savefig(outname+'_param'+str(j)+'chains.png') #Save figure, identifying which parameter
        plt.clf()
    return