import numpy as np
import time
from datetime import datetime,date
#from orbitclass import orbit
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

G = 4.*np.pi**2. #Useful constant
dt = 100./365.256 #Integration timestep: Adjust as desired, but beware inflating computation time

#Differential equation relating r, v, and a
def deriv_function(x0,G,M):
    #x0::x,y,z,vx,vy,vz
    x = x0[0]
    y = x0[1]
    z = x0[2]
    r = np.sqrt(x**2+y**2+z**2)
    vx = x0[3]
    vy = x0[4]
    vz = x0[5]
    ax = -G*M*x/r**3
    ay = -G*M*y/r**3
    az = -G*M*z/r**3
    #outputs: vx,vy,vz,ax,ay,az
    return np.array([vx,vy,vz,ax,ay,az])

#Custom 4th-order Runge Kutta function
#(since I didn't want to bother with finding one in scipy)
def RK_4(fun,x0,dt,*args):
    #Takes: function to integrate, initial position, timestep, function *args
    k1 = fun(x0,*args)
    k2 = fun(x0+dt/2.*k1,*args)
    k3 = fun(x0+dt/2.*k2,*args)
    k4 = fun(x0+dt*k3,*args)
    #Returns: Position at next timestep
    return x0+dt/6.*(k1+2*k2+2*k3+k4)

#Function to run an orbit from initial epoch to final epoch
#t0 and tf must be computed externally
def integration_loop(fun,t0,tf,x0,dt,G,M):
    t=t0
    ts = [t0]
    xs = [x0]
    #This is not a smart code. It just runs the loop until the time variable
    #reaches the final epoch. Thus be careful in choosing dt!
    while t<tf:
        xs.append(RK_4(fun,xs[-1],dt,G,M))
        t+=dt
        ts.append(t)
    return ts, xs

#the integration_loop() fn produces many evenly-spaced timesteps and positions.
#We interpolate between these to evaluate the actual position at the epochs where
#the planet has been observed.
def interpolate_astrometry(ts,xs,ep):
    full_int = interp1d(ts,xs,kind='cubic')
    return full_int(ep)
        
#Function used in the actual MCMC orbit fit. Takes epoch list & initial position,
#runs an integration from t0 to tf, then interpolates the trajectory to find
#positions at epochs of observations. Returns x,y position arrays corresponding to
#those epochs.
def interp_xy(fun,eps,x0,dt,G,M):
    t0 = np.min(eps)
    tf = np.max(eps)
    ts,xs = integration_loop(fun,t0,tf,x0,dt,G,M)
    #pdb.set_trace()
    full_int = interp1d(ts,xs,kind='cubic',axis=0)
    x_ret = []
    y_ret = []
    for ep in eps:
        xy = full_int(ep)
        x_ret.append(xy[0])
        y_ret.append(xy[1])
    return x_ret,y_ret

#Jacobian of deriv_function--UNUSED
def fun_jacob(x0,G,M):
    jacobian = np.zeros((6,6))
    x = x0[0]
    y = x0[1]
    z = x0[2]
    r = np.sqrt(x**2+y**2+z**2)
    vx = x0[3]
    vy = x0[4]
    vz = x0[5]
    jacobian[3,0]=1.
    jacobian[4,1]=1.
    jacobian[5,2]=1.
    jacobian[0,3]=G*M*deriv_match(x,r)
    jacobian[1,3]=G*M*deriv_cross(x,y,r)
    jacobian[2,3]=G*M*deriv_cross(x,z,r)
    jacobian[0,4]=G*M*deriv_cross(x,y,r)
    jacobian[1,4]=G*M*deriv_match(y,r)
    jacobian[2,4]=G*M*deriv_cross(y,z,r)
    jacobian[0,5]=G*M*deriv_cross(x,z,r)
    jacobian[1,5]=G*M*deriv_cross(y,z,r)
    jacobian[2,5]=G*M*deriv_match(z,r)
    return jacobian

#cross-term derivatives for jacobian--UNUSED
def deriv_cross(x1,x2,r):
    return 3.*x1*x2/(r**5.)

#diagonal-term derivatives for jacobian--UNUSED
def deriv_match(x1,r):
    return (1./(r**3))*(3*(x1**2)/(r**2)-1)

#Object's kinetic energy per unit mass, relative to host star
#(Literally just v^2)
def sp_kin_e(x0):
    vx = x0[3]
    vy = x0[4]
    vz = x0[5]
    v = np.sqrt(vx**2+vy**2+vz**2)
    return 0.5*(v**2)

#Object's potential energy per unit mass, relative to host star
def sp_pot_e(x0,G,M):
    x = x0[0]
    y = x0[1]
    z = x0[2]
    r = np.sqrt(x**2+y**2+z**2)
    return -G*M/r

#NOTE: UNUSED
#A good, adaptive 4th-order Runge Kutta would adjust the timestep to ensure the
#timestep is small enough to minimize errors but large enough to avoid bogging
#down the calculation. This fn was an ATTEMPT to check that timestep. Never
#finished implementing.
def check_timestep(fun,x0,dt,err_max,err_min,*args):
    result1 = RK_4(fun,x0,dt,*args)
    mid = RK_4(fun,x0,dt/2.,*args)
    result2 = RK_4(fun,mid,dt/2.,*args)
    diff = result1-result2
    error = 2*diff/(result1+result2)
    if np.max(np.abs(error))>err_max:
        return dt/np.sqrt(2)
    result3 = RK_4(fun,x0,dt*2,*args)
    result4 = RK_4(fun,result1,dt,*args)
    diff = result4-result3
    error = 2*diff/(result3+result4)
    if np.max(np.abs(error))<err_min:
        return dt*np.sqrt(2)
    return dt

