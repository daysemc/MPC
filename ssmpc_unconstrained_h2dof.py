'''
State Space Model Predictive Control (SSMPC) for Twin Rotor MIMO System (TRMS) / 2 DOF Helicopter
by Dayse

'''

print('Hello pie')

##### lib, packages e modules

import numpy as np
from numpy import rad2deg
import numpy.matlib
from numpy.linalg import matrix_power

from math import pi, radians, degrees, cos, sin
from control import tf, ss, tf2ss, step_response, sample_system, matlab

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

import matplotlib.pyplot as plt

##### system def & control config

### system def

# parameters of the system
# moment of inertia of helicopter body (kg-m^2)
L_body = 6.5*0.0254              # length of horizontal body (metal tube)
m_body = 0.094                   # mass of horizontal body (metal tube) 
J_body = m_body * L_body**2 / 12 # horizontal cylinder rotating about CM 
# moment of inertia of yoke fork that rotates about yaw aksis (kg-m^2)
m_yoke = 0.526                   # mass of entire yoke assembly (kg)
# h_yoke = 9*0.0254                # height of yoke assembly (m)
r_fork = 0.04/2                  # radius of each fork (approximated as cylinder)
J_yoke = 0.5 * m_yoke * r_fork**2
# moment of inertia from motor + guard assembly about pivot (kg-m^2)
m_prop = 0.43                    # mass of dc motor + shield + propeller shield
# m_motor = 0.203                  # mass of dc motor
r_prop = 6.25*0.0254             # distance from cm to center of pitch axis
J_prop = m_prop * r_prop**2      # using parallel axis theorem 
# equivalent moment of inertia about pitch and yaw axis (kg-m^2)
Jp = J_body + 2*J_prop           # pitch: body and 2 props
Jy = J_body + 2*J_prop + J_yoke  # yaw: body, 2 props, and yoke
# thrust-torque constant (N-m/V) [found experimentally]
Kpp = 0.0011                     # (pre-production unit: 0.0015)
Kyy = 0.0022                     # (pre-production unit: 0.0040)
Kpy = 0.0021                     # thrust acting on pitch from yaw (pre-production unit: 0.0020)
Kyp = -0.0027                    # thrust acting on yaw from pitch (pre-production unit: -0.0017) 
# stiffness (N-m/rad)[found experimentally] 
Ksp = 0.037463 
# viscous damping (N-m-s/rad) [found experimentally]
Dp = 0.0071116                   # pitch axis (pre-production unit: Dp = 0.0226) 
Dy = 0.0220                      # yaw axis (pre-production unit: Dy = 0.0211)

# state-space matrices
Ac = np.array([[0,0,1,0],[0,0,0,1],[-Ksp/Jp,0,-Dp/Jp,0],[0,0,0,-Dy/Jy]]) # state/system matrix, Rnxn
Bc = np.array([[0,0],[0,0],[Kpp/Jp,Kpy/Jp],[Kyp/Jy,Kyy/Jy]])             # input/control matrix, Rnxp
Cc = np.array([[1,0,0,0],[0,1,0,0]])                                     # output matrix, Rqxn (usually identity)
Dc = np.array([[0,0],[0,0]])                                             # feedthrought/feedforward matrix, Rqxp (usually zero)

### state-sapace representation & step response

## continuous-time linear state-space representation
sysc = ss(Ac,Bc,Cc,Dc) 

## dimensions
n = len(Ac) # number of states (=A rows)
p = len(Bc[0]) # number of control actions (=B columns)
q = len(Cc) # number of outputs (=C rows)

# ## uncomment to analyze step response
# # time vector
# t = 50
# tvec = np.arange(0,t,1)

# # data
# _, yout0 = step_response(sysc,T=tvec,input=0) # data 0
# _, yout1 = step_response(sysc,T=tvec,input=1) # data 1

# # plot
# plt.figure()
# for i in range(0,2):
#     plt.subplot(2,2,i+1)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout0[i]), label='input=0') 
# # end for
# for i in range(0,2):
#     plt.subplot(2,2,i+3)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout1[i]), label='input=1')
# #end for
# plt.show() 

### system "discretization"

## time config
fsim = 10.0    # simulation frequency (ex: 100)
Ta  = 30.0     # settling time (from step response) (ex: 50)
Ts  = Ta/40.0  # sample time
tsim = 750.0   # simulation time (ex: 300s = 5min)
Te = Ts/fsim   # emulation time

## discrete-time state-space model 
sysd = sample_system(sysc,Ts,method='zoh')
Ad = sysd.A
Bd = sysd.B
Cd = sysd.C
Dd = sysd.D

## discrete-time state-space model for emulation
syse = sample_system(sysc,Te,method='zoh')
Ae = syse.A
Be = syse.B
Ce = syse.C
De = syse.D

### mpc parameters (from tests)

N  = 3     # prediction horizon (ex: 3)
M  = 3     # control horizon (ex: 3)
py = 1     # output weight (ex: 1)
pu = 0.001 # control weight (ex: 0.001)

## prediction model with integral action
Aksi = np.block([[Ad,np.zeros((n,q))],[Cd@Ad,np.identity(q)]])
Bksi = np.block([[Bd],[Cd@Bd]])
Cksi = np.block([np.zeros((q,n)),np.identity(q)]) 

## G (dynamic matrix)
Gaux = Cksi@Bksi
for i in range(1,N):
    Gaux = np.block([[Gaux],[Cksi@matrix_power(Aksi,i)@Bksi]])
# end for
G = Gaux
if M > 1:
    for i in range(1,M):
        G = np.block([G,np.block([[np.zeros((q*i,p))],[Gaux[:((N*q)-(q*i)),:]]])])
    # end for
# end if

# phi
phi = Cksi@Aksi
for i in range(2,N+1):
    phi = np.block([[phi],[Cksi@matrix_power(Aksi,i)]])
# end for

## Py (output weight)
Py = py*np.identity(q*N)

## Pu (control weight, != 0)
Pu = pu*np.identity(p*M)

## Kmpc
kaux = np.linalg.inv((G.transpose()@Py@G) + Pu)@G.transpose()@Py
kmpc = kaux[:p,:]

##### control system

### init config

## sim parameters
ref1 = radians(30) # pitch ref
ref2 = radians(30) # yaw ref
tsim = 23          # sim time

## sim parameters
step = Te       # sim steps
ts = Ts         # sampling period
nit = tsim/step # total number of simulation points
fsim = ts/step  # number of points awaiting sampling
typlot = np.arange(0,tsim,step)      # y plot time
tuplot = np.arange(0,tsim-step,step) # u plot time
## plant/emulation time
ke = 0

## controller time
kc = 0 

## output vector, Rqx1
y = np.zeros((len(np.arange(0,tsim,step)),q))

## state vector, Rnx1
x = np.zeros((len(np.arange(0,tsim,step)),n))

## last x
xpast = np.zeros((1,n))

## Δx
dx = np.zeros((1,n))

## input/control vector, Rpx1, in kc
u1 = np.zeros((len(np.arange(0,tsim,step)),p)) 

## input/control vector, Rpx1, in ke
u2 = np.zeros((len(np.arange(0,tsim,step)),p))

## Δu
du = np.zeros((len(np.arange(0,tsim,step)),p))

## ref vector
r = np.zeros((q*N,1))
for i in range(0,q*N,q):
    r[i] = ref1
    r[i+1] = ref2
# end for
ref = r

## ksi
ksi = np.zeros((1,q+n))

### sim loop
for t in np.arange(step,tsim-step,step):

    # increment the logic counter
    ke = ke + 1 

    # ## edit to create a variable reference
    # if ke>nit/3 and ke<2*nit/3:
    #     r = ref*2
    # else:
    #     r = ref
    # # end if-else

    ### control loop
    if ke==0 or ke%fsim==0:

        ## kc counting
        kc = kc + 1

        ## ksi 
        dx = x[ke] - xpast         # update dx
        xpast = x[ke]              # save past x & change var name
        ksi = np.block([dx,y[ke]]) # obtain ksi
        ksi = ksi.reshape(q+n,1)   # reshape ksi into a column

        # free sys response
        f = phi@ksi 

        # Δu
        du = kmpc@(r-f) 
        du = du.reshape(1,p) # reshape Δu into a row

        # u
        if kc==1:
            u1[kc] = du
        else:
            u1[kc] = u1[kc-1] + du
        # end if-else
    
    ## sim real sys
    x[ke+1] = Ae@x[ke] + Be@u1[kc]
    y[ke+1] = Ce@x[ke+1]
    u2[ke] = u1[kc]

### plot

# break the output y into two distinct variables
yplot1 = y[:,0]                                        # save 1st column values in y1 vector
yplot1 = yplot1.reshape(len(np.arange(0,tsim,step)),1) # reshape y1 into a column vector
yplot1 = rad2deg(yplot1)                               # convert measuring unit from radian to degree
yplot2 = y[:,1]                                        # save 2nd column values in y2 vector
yplot2 = yplot2.reshape(len(np.arange(0,tsim,step)),1) # reshape y2 into a column vector
yplot2 = rad2deg(yplot2)                               # convert measuring unit from radian to degree

# break the control action u into two distinct variables
uplot1 = u2[:len(np.arange(0,tsim,step))-1,0]            # save 1st column values in u1 vector
uplot1 = uplot1.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u1 into a column vector
uplot2 = u2[:len(np.arange(0,tsim,step))-1,1]            # save 2nd column values in u2 vector
uplot2 = uplot2.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u2 into a column vector

# figure
plt.figure()

# subplot 1
plt.subplot(2,1,1)
plt.plot(typlot,yplot1,label='Pitch')
plt.plot(typlot,yplot2,label='Yaw') 

plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend(loc=4)
plt.grid(linestyle=':')

# subplot 2
plt.subplot(2,1,2)
plt.plot(tuplot,uplot1,label="Main rotor")
plt.plot(tuplot,uplot2,label="Tail rotor")

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(loc=4)
plt.grid(linestyle=':')

# show
plt.show() 