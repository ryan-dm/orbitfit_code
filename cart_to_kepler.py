import numpy as np
from RK4_energy import sp_kin_e,sp_pot_e
import matplotlib.pyplot as plt
import pdb
from orbitclass import orbit

#Inputs: cartesian MC parameter vector & epoch at which vector applies
#        Must have elements 0 through 6 be state vector & Mass
#Outputs: Keplerian parameter vector
def cart_to_kepler(theta,ep):

    G = 4*np.pi**2 #Always need this constant

    #Unpack cartesian state vector
    x = theta[0]
    y = theta[1]
    z = theta[2]
    vx = theta[3]
    vy = theta[4]
    vz = theta[5]
    #Also separate out state vector from mass
    x0 = theta[:6]
    M = theta[6]

    #Create r,v vectors and magnitudes
    r_vec = theta[0:3]
    v_vec = theta[3:6]
    r = mag(r_vec)
    v = mag(v_vec)
    mu = G*M #Useful constant

    #Compute eccentricity vector...and its magnitude is the eccentricity
    e_vec = ((v**2)/mu - 1./r)*r_vec - (dot(r_vec,v_vec)/mu)*v_vec
    e = mag(e_vec)

    #Compute specific angular momentum and specific energy
    #L is perpendicular to the plane of the orbit--computationally useful
    #Specific energy is also computationally useful, and straightforwardly
    #relates to eccentricity
    Lsp = cross(r_vec,v_vec)
    Lsp_mag = mag(Lsp)
    Esp = sp_kin_e(x0)+sp_pot_e(x0,G,M)

    #Alternative way to compute eccentricity...gives same value as e
    e2 = np.sqrt((2.*mag(Lsp)**2)*Esp/mu**2+1.)

    #For non-parabolic orbits: compute semimajor axis and periastron
    if e!=1:
        a = 1./(2./r - (v**2)/mu)
        q = a*(1-e)

    #Inclination is the angle between the orbital plane and the sky plane
    #i.e. the angle between the normal vectors to these two planes
    #i.e. the angle between angular momentum vector and the negative z axis
    inc = np.arccos(dot(Lsp,[0,0,-1])/mag(Lsp))

    #negative z axis is normal to sky plane
    #Angular momentum is normal to orbital plane
    #Their cross product is orthogonal to both vectors--meaning, it lies
    #In both the orbital plane, and in the sky plane
    #This is therefore an expression for the line of nodes
    parallel_intersect = cross(Lsp,[0,0,-1])

    #Longitude of ascending node: first euler angle
    #Defines the axis about which the inclination rotation is applied (i.e.
    # the line of nodes)
    xlan = parallel_intersect[0]
    ylan = parallel_intersect[1]

    #Ascending node is where planet moves from -z to +z crossing the plane of the sky
    lan1 = np.arctan2(xlan,ylan)%(2*np.pi)
    lan2 = (lan1+np.pi)%(2*np.pi) #NOTE: This is properly the longitude of the DESCENDING node

    #Create a vector (r_perp) that is perpendicular to the radial vector,
    #While still in the orbital plane (i.e. also perpendicular to the angular momentum)
    #This produces a vector in ROUGHLY the same direction as the velocity
    r_perp = cross(Lsp,r_vec)
    #Calculate phi, the angle between r_perp and velocity, and use it to compute
    #the current value of the true anomaly -- only used for parabolic orbits
    phi = np.arccos(dot(v_vec,r_perp)/mag(r_perp)/mag(v_vec))
    #The angle b/w radial vector and velocity vector is deterministic

    dotrv = dot(r_vec,v_vec)


    #For non-parabolic orbits: Compute true anomaly since r and e are known
    if e!=1:
        p = q*(1+e)
        TA = (np.arccos((p/r-1.)/e))%(2*np.pi) #True anomaly
        if dotrv<0: TA = 2*np.pi-TA #Correct angle wrapping
    else: #For parabolas, true anomaly is exactly double phi
        TA = 2*phi 
        q = (k/v**2.)*(1+np.cos(TA)) #Haven't computed parabolic periastron yet, do so here
        

    #omega:
    #TA is an in-orbital-plane rotation from omega
    #We have LAN and inc, defining the angle & axis to rotate
    #the orbital plane back to the sky plane (or, equivalently,
    # to rotate the default coordinate system into an in-orbital-plane
    # coordinate system; in this coordinate system, the angle b/w the
    # transformed y-axis and the data point == TA + omega)
    # so: rotate [x,y,z] by LAN and inc, calc. angle from y-axis; angle = TA + omega
    
    #Rotate (x,y,z) by LAN and inc to enter in-orbital-plane coordinates
    rotated_pos = rotate_i_o(x,y,z,inc,lan1)
    # Find the position angle of thep lanet in these coordinates
    rot_PA = np.arctan2(rotated_pos[0],rotated_pos[1])
    # Compute argument of periastron from true anomaly and orbital-plane position angle
    argp = (rot_PA-TA)%(2*np.pi)


    #tau:
    # TA transforms to EA transforms to MA (transforms to delta-t)
    # subtract this from the epoch of the observation and get tau
    #NOTE: I use different variables (EA, DA, FA) for the eccentric anomaly for
    # e<1, e=1, and e>1, since they're all computed somewhat differently.
    # EA and FA can be cross-converted by correct multiplication by sqrt(-1), but
    # the parabolic DA variable is mostly a computational tool
    TA = (TA+np.pi)%(2*np.pi)-np.pi #Ensure true anomaly is within [-pi,pi]
    if e==0: #Circular orbit: All anomalies equal
        MA = TA%(2*np.pi)
    elif e<1: #elliptical orbit: normal trig functions
        EA = np.arccos((e+np.cos(TA))/(1.+e*np.cos(TA)))
        if TA<0: EA*=-1
        MA = EA-e*np.sin(EA) #Easy to solve going THIS direction

    elif e==1: #Parabolic orbit: "eccentric" anomaly is deterministic
        DA = np.tan(TA/2.) #not the normal representation but easy for conversions
        MA = DA+(DA**3.)/3. #Mean anomaly (linear w/ time) is cubic w/ "eccentric" anomaly
    else: #Hyperbolic orbit: hyperbolic trig functions
        FA = np.arccosh((e+np.cos(TA))/(1.+e*np.cos(TA)))
        if TA<0: FA*=-1
        MA = e*np.sinh(FA)-FA

    #Non-parabolic: compute 1/(mean motion) with semimajor axis
    if e!=1:
        nuinv = ((np.abs(a)**3.)/mu)**(0.5)
    else: #Parabolic: semimajor axis undefined, use periastron
        nuinv = ((2.*(q**3.))/mu)**(0.5)

    #Convert mean anomaly into delta-time
    dt = nuinv*MA*365.256
    tau = ep - dt #Tau is difference between observed epoch and 
    if e<1: #For bound orbits, wrap tau by period
        P = 2*np.pi*nuinv*365.256
        tau = (tau-ep)%P+ep-P
    return q,tau,argp,lan1,inc,e

#Custom cross product function for convenience
def cross(u,v):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    w1 = u2*v3-u3*v2
    w2 = u3*v1-u1*v3
    w3 = u1*v2-u2*v1
    return np.array([w1,w2,w3])

#Custom 2-angle rotation for convenience
#Note that i and o (inclination and Omega, longitude of ascending node)
#MUST be provided in a specific order!
def rotate_i_o(x,y,z,i,o):
    x1 = x*np.cos(o)-y*np.sin(o)
    y1 = x*np.sin(o)+y*np.cos(o)
    z1 = z
    x2 = x1*np.cos(i)+z1*np.sin(i)
    y2 = y1
    z2 = -x1*np.sin(i)+z1*np.cos(i)
    return np.array([x2,y2,z2])

#Custom dot product for convenience
def dot(u,v):
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]

#Custom vector magnitude for convenience
def mag(u):
    return np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
