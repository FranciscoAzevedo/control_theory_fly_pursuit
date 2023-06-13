#  Code adapted from here
# https://apmonitor.com/pdc/index.php/Main/CourseSchedule

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
%matplotlib qt

fd_path = '/home/paco-laptop/Desktop/paco/code/control_theory_pursuit/'

# %% Tank level controller

# simulation config
t = 10 # seconds
n_updates = 101

# animate plots?
animate=False # True / False

# define tank model
def tank(Level,time,c,valve):
    rho = 1000.0 # water density (kg/m^3)
    A = 1.0      # tank area (m^2)
    # calculate derivative of the Level
    dLevel_dt = (c/(rho*A)) * valve
    return dLevel_dt

# time span for the simulation for 10 sec, every 0.1 sec
ts = np.linspace(0,t,n_updates)

# valve operation
c = 50.0          # valve coefficient (kg/s / %open)
u = np.zeros(n_updates) # u = valve % open

# level initial condition
PV = 0

# initial valve position
valve = 10 # only 10 percent opened

# for storing the results
z = np.zeros(n_updates)
es = np.zeros(n_updates) # errors

# TO DO: what is the value for ubias?
ubias = u[0]

# TO DO: record the desired level (set point)
SP = np.zeros(n_updates)
SP[20:] = 10

# %% Proportional controller without dead time

# TO DO: decide on a tuning value for Kp
Kp = 100 # Not done via FOPDT but will be playing around with it

fig = plt.figure(1,figsize=(12,5))
ax = fig.add_subplot(211)

if animate:
    plt.ion()
    plt.show()

# simulate with ODEINT
for i in range(100):
    # calculate the error
    es[i] = SP[i] - PV

    # TO DO: put P-only controller here
    valve = ubias + Kp*es[i]

    # Clipping max values achievable
    valve = max(0,valve)
    valve = min(100,valve)

    u[i+1] = valve   # store the valve position
    y = odeint(tank,PV,[0,0.1],args=(c,valve))
    PV = y[-1] # take the last point
    z[i+1] = PV # store the PV for plotting

    if animate:
        # update plot
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(ts[0:i+1],z[0:i+1],'r-',linewidth=3,label='level PV')
        plt.plot(ts[0:i+1],SP[0:i+1],'k:',linewidth=3,label='level SP')
        plt.ylabel('Tank Level')
        plt.legend(loc='best')
        plt.subplot(3,1,2)
        plt.plot(ts[0:i+1],u[0:i+1],'b--',linewidth=3,label='valve')
        plt.ylabel('Valve')    
        plt.legend(loc='best')
        plt.subplot(3,1,3)
        plt.plot(ts[0:i+1],es[0:i+1],'g-',linewidth=3,label='error')
        plt.ylabel('Error = SP-PV')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.pause(0.1)

if not animate:
    # plot results
    plt.subplot(3,1,1)
    plt.plot(ts,z,'r-',linewidth=3,label='level PV')
    plt.plot(ts,SP,'k:',linewidth=3,label='level SP')
    plt.ylabel('Tank Level')
    plt.legend(loc='best')
    plt.subplot(3,1,2)
    plt.plot(ts,u,'b--',linewidth=3,label='valve')
    plt.ylabel('Valve')    
    plt.legend(loc='best')
    plt.subplot(3,1,3)
    plt.plot(ts,es,'g-',linewidth=3,label='error')
    plt.ylabel('Error = SP-PV')    
    plt.xlabel('Time (sec)')
    plt.legend(loc='best')
    plt.show()

# %% Proportional Integral Controller without dead times (delay)

# define tank model WITH leak
def tank(Level,time,c,valve):
    rho = 1000.0 # water density (kg/m^3)
    A = 1.0      # tank area (m^2)
    
    leak_cross_sec = 30
    # calculate derivative of the Level
    dLevel_dt = (c/(rho*A)) * valve - (c/(rho*A)) * leak_cross_sec
    return dLevel_dt

# Controller parameters
Kp = 10
Ki = 10
int_win_size = 10

# pre-allocating variables
ie = np.zeros(n_updates) # integral error
PVs = np.zeros(n_updates) # process variable
es = np.zeros(n_updates) 
u = np.zeros(n_updates) # u = valve % open

fig,axes = plt.subplots(1,figsize=(12,5))

PV = 0
# simulate with ODEINT
for i in range(100):
    # calculate the error
    es[i] = SP[i] - PV

    # integrate of last "int_win_size" errors
    ie[i] = np.sum(es[(i-int_win_size):i])

    # PI controller
    valve = ubias + Kp*es[i] + Ki*ie[i]

    # Clipping max values achievable
    valve = max(0,valve)
    valve = min(100,valve)

    u[i] = valve   # store the valve position
    
    y = odeint(tank,PV,[0,0.1],args=(c,valve))
    PV = y[-1] # take the last point

    PVs[i] = PV

# plot results
plt.subplot(3,1,1)
plt.title('Params: Kp='+str(Kp) +' Ki='+str(Ki) +' int_win_size='+str(int_win_size))
plt.plot(ts,PVs,'r-',linewidth=3,label='level PV')
plt.plot(ts,SP,'k:',linewidth=3,label='level SP')
plt.ylabel('Tank Level')
plt.legend(loc='best')

plt.subplot(3,1,2)
plt.plot(ts,u,'b--',linewidth=3,label='valve')
plt.ylabel('Valve')    
plt.legend(loc='best')

plt.subplot(3,1,3)
plt.axhline(0,c = 'r')
plt.plot(ts,es,'g-',linewidth=3,label='error')
plt.ylabel('Error = SP-PV')    
plt.xlabel('Time (sec)')
plt.legend(loc='best')
plt.show()

fig.savefig(fd_path + 'figures/PI_' + 'Kp='+str(Kp) +'_Ki='+str(Ki) +'_int_win_size='+str(int_win_size))
# %% Full PID
Kp = 10
Ki = 20
Kd = 50
int_win_size = 5

# pre-allocating variables
PVs = np.zeros(n_updates) # process variable
es = np.zeros(n_updates) # errors
ops = np.zeros(n_updates) # u = valve % open
Ps = np.zeros(n_updates) 
Is = np.zeros(n_updates) 
Ds = np.zeros(n_updates)

# two bumps
#SP[50:80] = 0

fig, axes = plt.subplots(1,figsize=(12,8))

def pid(sp,pvs,iter):

    e = sp-pvs[iter-1]
    ie = np.sum(es[(iter-int_win_size):iter])

    if iter > 10:
        dpv = pvs[iter-1] - pvs[iter-5]
    else:
        dpv = 0

    P = Kp*e
    I = Ki*ie
    D = -Kd*dpv

    # PI controller
    op = ubias + P + I + D

    # Clipping max values achievable
    op = max(0,op)
    op = min(100,op)

    return [op,e,P,I,D]

PV = 0
for i in range(100):
    op, e, P,I,D = pid(SP[i], PVs, i)
    
    y = odeint(tank,PV,[0,0.1],args=(c,op))
    PV = y[-1] # take the last point
    
    #  store vars
    es[i] = e
    ops[i] = op
    PVs[i] = PV
    Ps[i] = P
    Is[i] = I
    Ds[i] = D

# plot results
plt.subplot(4,1,1)
plt.title('Params:' + 'Kp='+str(Kp) +' Ki='+str(Ki) +' int_win_size='+str(int_win_size) +' Kd='+str(Kd))
plt.plot(ts,PVs,'r-',linewidth=3,label='level PV')
plt.plot(ts,SP,'k:',linewidth=3,label='level SP')
plt.ylabel('Tank Level')
plt.legend(loc='best',frameon=False)

plt.subplot(4,1,2)
plt.plot(ts,ops,'b--',linewidth=3,label='valve')
plt.ylabel('Valve')    
plt.legend(loc='best',frameon=False)

plt.subplot(4,1,3)
plt.axhline(0,c = 'r')
plt.plot(ts,es,'g-',linewidth=3,label='error')
plt.ylabel('Error = SP-PV')    
plt.xlabel('Time (sec)')
plt.legend(loc='best', frameon=False)

plt.subplot(4,1,4)
plt.plot(ts,Ps,label='Prop')
plt.plot(ts,Is,label='Int')
plt.plot(ts,Ds,label='Der')
plt.ylabel('Controllers effort')    
plt.xlabel('Time (sec)')
plt.legend(loc='best',frameon=False)

plt.show()

fig.savefig(fd_path + 'figures/PID_' + 'Kp='+str(Kp) +'_Ki='+str(Ki) +'_int_win_size='+str(int_win_size) +'_Kd='+str(Kd))

# %% Things to implement for the future
# 1 - Dead time between input and output
# 2 - Low pass on derivative to reduce HF oscillations on output
# 3 - Reset windup on the integral component