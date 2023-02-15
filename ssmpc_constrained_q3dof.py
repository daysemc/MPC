'''
State Space Model Predictive Control (SSMPC) for Quadcopter / 3 DOF Hover
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
Ktn = 0.0036
Ktc = -0.0036
Kf = 0.1188
l = 0.197
Jy = 0.110
Jp = 0.0552
Jr = 0.0552

# state-space matrices
Ac = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
Bc = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[Ktc/Jy,Ktc/Jy,Ktn/Jy,Ktn/Jy],[l*Kf/Jp,-l*Kf/Jp,0,0],[0,0,l*Kf/Jr,-l*Kf/Jr]])
Cc = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
Dc = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])

### state-sapace representation & step response

## continuous-time linear state-space representation
sysc = ss(Ac,Bc,Cc,Dc) 

## dimensions
n = len(Ac) # number of states (=A rows)
p = len(Bc[0]) # number of control actions (=B columns)
q = len(Cc) # number of outputs (=C rows)

# ## uncomment to analyze step response
# # time vector
# tsim = 50
# tvec = np.arange(0,tsim,1)

# # data
# _, yout0 = step_response(sysc,T=tvec,input=0) # data 0
# _, yout1 = step_response(sysc,T=tvec,input=1) # data 1
# _, yout2 = step_response(sysc,T=tvec,input=2) # data 2
# _, yout3 = step_response(sysc,T=tvec,input=3) # data 3

# # plot
# plt.figure()
# for i in range(0,3):
#     plt.subplot(4,3,i+1)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout0[i]), label='input=0') 
# # end for
# for i in range(0,3):
#     plt.subplot(4,3,i+4)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout1[i]), label='input=1')
# #end for
# for i in range(0,3):
#     plt.subplot(4,3,i+7)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout2[i]), label='input=2')
# #end for
# for i in range(0,3):
#     plt.subplot(4,3,i+10)
#     plt.grid(True)
#     plt.plot(tvec, np.transpose(yout3[i]), label='input=3')
# #end for
# plt.show() 

### system "discretization"

## time config
fsim = 10.0    # simulation frequency (ex: 100)
Ta  = 30.0     # settling time (from step response) (ex: 50)
Ts  = Ta/40.0 # sample time
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

## prediction model with input increments
Aksi = np.block([[Ad,Bd],[np.zeros((p,n)),np.identity(p)]])
Bksi = np.block([[Bd],[np.identity(p)]])
Cksi = np.block([Cd,np.zeros((q,p))])  

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

### constraints handling

## Gn
Gn = Py@G

## Hqp
Hqp = 2*((G.transpose()@Py@G) + Pu)

## I
IpM = np.identity(p*M)

## T
Ip1 = np.matlib.repmat(np.identity(p),M,1)
TM = Ip1
if M > 1: # TMp
    for i in range(1,M):
        TM = np.block([TM,np.block([[np.zeros((p*i,p))],[Ip1[:((M*p)-(p*i)),:]]])])
    # end for
# end if

## Aqp = S
Aqp = np.vstack((IpM,-IpM,TM,-TM,G,-G))

##### control system

### init config

## sim parameters
ref1 = radians(10) # yaw
ref2 = radians(10) # pitch
ref3 = radians(10) # roll
tsim = 23          # sim time

## sys contraints (datasheet)
# lim y: yaw -360 ~ +360º, pitch -15 ~ +15º, roll -15 ~ +15º
ymin = np.array([[radians(-360.0)],[radians(-30.0)],[radians(-30.0)]]) 
ymax = np.array([[radians(360.0)],[radians(30.0)],[radians(30.0)]])
# lim u: -24 ~ +24V
umin = ([[0],[0],[0],[0]])
umax = np.array([[12],[12],[12],[12]]) 
# lim Δu: 3V
dumin = np.array([[-1.5],[-1.5],[-1.5],[-1.5]])
dumax = np.array([[1.5],[1.5],[1.5],[1.5]])

## sim parameters
step = Te       # sim steps
ts = Ts         # sampling period
nit = tsim/step # total number of simulation points
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
    r[i+2] = ref3
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
    if ke==1 or ke%fsim==0:

        ## kc counting
        kc = kc + 1

        ## ksi 
        ksi = np.block([x[ke],u2[ke-1]]) # obtain ksi
        ksi = ksi.reshape(n+p,1)         # reshape ksi into a column

        # free sys response
        f = phi@ksi 

        ### quadratic programming
        
        ## quadprog f
        fqp = 2*Gn.transpose()@(f-r) 

        try:
            ## quadprog b
            if kc == 1:
                    bqp = np.vstack((np.matlib.repmat(dumax,M,1),-np.matlib.repmat(dumin,M,1),np.matlib.repmat((umax-np.zeros((p,1))),M,1),np.matlib.repmat((np.zeros((p,1))-umin),M,1),(np.matlib.repmat(ymax,N,1)-f),(f-np.matlib.repmat(ymin,N,1))))
            else:
                bqp = np.vstack((np.matlib.repmat(dumax,M,1),-np.matlib.repmat(dumin,M,1),np.matlib.repmat((umax-u1[kc-1].reshape((p,1))),M,1),np.matlib.repmat((u1[kc-1].reshape((p,1))-umin),M,1),(np.matlib.repmat(ymax,N,1)-f),(f-np.matlib.repmat(ymin,N,1))))
            # end if-else

            ## solution to the opt problem
            sol = solvers.qp(matrix(Hqp),matrix(fqp),matrix(Aqp),matrix(bqp)) 
            if sol['status'] == 'unknown':
                raise ValueError()
            # end if

            ## Δu
            du = np.asarray(sol['x'])
            du = du[:p].reshape(1,p) # reshape Δu into a row

        except ValueError: # no feasible solution
            du  = np.zeros((1,p))
        # end try-except

        ## u
        if kc == 1:
            u1[kc] = du
        else:
            u1[kc] = u1[kc-1] + du
        # end if-else

    ## sim real sys
    x[ke+1] = Ae@x[ke] + Be@u1[kc]
    y[ke+1] = Ce@x[ke+1]
    u2[ke] = u1[kc]

### plot

# break the output y into 3 distinct variables
yplot1 = y[:,0]                                        # save 1st column values in y1 vector
yplot1 = yplot1.reshape(len(np.arange(0,tsim,step)),1) # reshape y1 into a column vector
yplot1 = rad2deg(yplot1)                               # convert measuring unit from radian to degree
yplot2 = y[:,1]                                        # save 2nd column values in y2 vector
yplot2 = yplot2.reshape(len(np.arange(0,tsim,step)),1) # reshape y2 into a column vector
yplot2 = rad2deg(yplot2)                               # convert measuring unit from radian to degree
yplot3 = y[:,2]                                        # save 3rd column values in y3 vector
yplot3 = yplot3.reshape(len(np.arange(0,tsim,step)),1) # reshape y3 into a column vector
yplot3 = rad2deg(yplot3)                               # convert measuring unit from radian to degree


# break the control action u into two distinct variables
uplot1 = u2[:len(np.arange(0,tsim,step))-1,0]            # save 1st column values in u1 vector
uplot1 = uplot1.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u1 into a column vector
uplot2 = u2[:len(np.arange(0,tsim,step))-1,1]            # save 2nd column values in u2 vector
uplot2 = uplot2.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u2 into a column vector
uplot3 = u2[:len(np.arange(0,tsim,step))-1,2]            # save 3rd column values in u3 vector
uplot3 = uplot1.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u3 into a column vector
uplot4 = u2[:len(np.arange(0,tsim,step))-1,3]            # save 4th column values in u4 vector
uplot4 = uplot2.reshape(len(np.arange(0,tsim,step))-1,1) # reshape u4 into a column vector

# figure
plt.figure()

# subplot 1
plt.subplot(2,1,1)
plt.plot(typlot,yplot1,linestyle="-",label='Yaw')    # yaw
plt.plot(typlot,yplot2,linestyle="--",label='Pitch') # pitch
plt.plot(typlot,yplot3,linestyle=":",label='Roll')   # roll

plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend(loc=4)
plt.grid(linestyle=':')

# subplot 2
plt.subplot(2,1,2)
plt.plot(tuplot,uplot1,linestyle="-",label="Front rotor")
plt.plot(tuplot,uplot2,linestyle="--",label="Back rotor")
plt.plot(tuplot,uplot3,linestyle="-.",label="Right rotor")
plt.plot(tuplot,uplot4,linestyle=":",label="Left rotor")

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(loc=4)
plt.grid(linestyle=':')

# show
plt.show() 