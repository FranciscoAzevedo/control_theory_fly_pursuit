"""
    Rotation work for Eugenia Chiappe May-June 2023
    Author: Francisco Moreira de Azevedo, 2023
    e-mail: francisco.azevedo@research.fchampalimaud.org
"""

# %% Imports
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import imageio
import os
%matplotlib qt

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.linewidth': 2,
         'figure.dpi': 120,
         }

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams.update(params)

fd_path = '/home/paco-laptop/Desktop/paco/code/control_theory_fly_pursuit/'

# %% Helper functions

# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

# Define function to compute lambda
def get_range_vec(pursuer_pos,target_pos):
    range_vec_unnorm = target_pos-pursuer_pos
    range_vec = range_vec_unnorm / np.linalg.norm(range_vec_unnorm) # unit vector
    return np.array(range_vec)

def get_angle_atan2(comp_vec, ref_vec):
    vec = comp_vec-ref_vec
    dX = vec[0]
    dY = vec[1]
    rad = math.atan2(dY,dX)
    return rad

def wrap_angles(rads):
    wrapped_rads = (rads + np.pi) % (2 * np.pi) - np.pi
    return wrapped_rads

# computing velocity
def get_velocity(XY_pos):

    XY_pos_shift_fwd = np.roll(XY_pos,-1) # shift one array forward

    # compute difference and square
    # does both subtraction and power operations to X and Y seperately
    pos_diffs = (XY_pos_shift_fwd-XY_pos)**2 

    norms = np.sqrt(pos_diffs[0,:]+pos_diffs[1,:]) # compute norm
    
    return np.array(norms) # last is always wrong in this implementation..

# %% Define PID controller
def pid(iter,es,pvs, Kp=1,Ki=0,Kd=0,int_win_size=0):

    """
        iter: iteration number 
        es: error
        pvs: process variables
        Kp,Ki,Kd,int_win_size: PID controller weights and settings
    """

    # integral windup reset not implemented
    if iter < 1:
        ie = 0
    elif (iter < int_win_size) & (iter >= 1):
        ie = np.mean(es[:iter])
    else:
        ie = np.mean(es[(iter-int_win_size):iter])

    # avoid crashing due to not having a derivative in the first iter
    if iter > 1:
        # derivative on PV instead of error to avoid derivative kick
        dpv = (pvs[iter-1]-pvs[iter])
    else:
        dpv = 0

    # parallel implemention
    P = Kp*es[iter]
    I = Ki*ie
    D = -Kd*dpv 

    # PID controller
    op = pvs[iter] + P + I + D 

    return [op,P,I,D]

# %% Setting up playground and trajectory of target

# Generic variables

# lead_lag<0 means delayed, >0 means looking ahead
lead_lag = 0 
# coordinate referential
ref_vec = np.array([0.01,0])

# Defines whether you load an experimental path or pre-programmed
exp = False
data_path = 'parallel.npy'

if exp == True:
    data = np.load(data_path)
    sps = data[0:2,:]

    # simulation config
    t = data[2,-1] - data[2,0] # full duration
    n_updates = len(data[2,:])
    ts = np.linspace(0,t,n_updates)

    # playground dimensions
    dim_x = max(abs(sps[0,:]))*2
    dim_y = max(abs(sps[1,:]))*2

    # Filling up nans with previous value registered
    nans, x = nan_helper(sps[0,:])
    sps[0,:][nans]= np.interp(x(nans), x(~nans), sps[0,:][~nans])

    nans, x = nan_helper(sps[1,:])
    sps[1,:][nans]= np.interp(x(nans), x(~nans), sps[1,:][~nans])

    # chaning referential to lower left instead of center
    sps[0,:] = sps[0,:] + dim_x/2
    sps[1,:] = sps[1,:] + dim_y/2

elif exp == False:
    # simulation config
    t = 10 # seconds
    n_updates = 100
    ts = np.linspace(0,t,n_updates)

    # playground dimensions
    dim_x = 200
    dim_y = 300

    # pre-programmed paths (MUST BEGIN AND START IN SAME POINT!!!)

    # Define path of the target (Set Point _is_ the target position)
    sps = np.zeros((2, n_updates))
    traj =  'straight'

    # square wave l2r, to test int windup reset
    if traj == 'square_wave':
        t = np.linspace(0,dim_x, num=n_updates) 
        freq = 1 # Hz?
        sps[0,:] = t 
        sps[1,:] = signal.square(2 * np.pi * freq * t)*20 + dim_y/2

    # straight line left to right
    if traj == 'straight':
        sps[0,:] = np.linspace(0,dim_x, num=n_updates)  
        sps[1,:] = dim_y/2

    # line with jitter noise
    if traj == 'jitter_line':
        sps[0,:] = np.linspace(0,dim_x, num=n_updates)  

        # jitter
        noise_sigma = 3
        sps[1,:] = np.random.normal(dim_y/2,noise_sigma,n_updates)
        
    # circle
    elif traj == 'circle':
        thetas = np.linspace(0 , 4*np.pi , n_updates)
        radius = 100
        offset_x, offset_y = dim_x/2,dim_y/2
        for i,theta in enumerate(thetas):
            sps[0,i] = radius * np.cos( theta ) + offset_x # x dimension
            sps[1,i] = radius * np.sin( theta ) + offset_y # y dimension

    # square
    elif traj == 'square':
        edge = int(min(dim_y,dim_x)/6) # 6 is completely arbitrary but fits
        v_len = dim_y-2*edge
        h_len = dim_x-2*edge

        ll_to_ul = np.vstack((np.ones(v_len)*edge,np.arange(edge,dim_y-edge)))
        ul_to_ur = np.vstack((np.arange(edge,dim_x-edge), np.ones(h_len)*(dim_y-edge)))

        ur_to_lr = np.vstack((np.ones(v_len)*(dim_x-edge),np.arange(dim_y-edge,edge,-1)))
        lr_to_ll = np.vstack((np.arange(dim_x-edge,edge,-1), np.ones(h_len)*(edge)))

        path = np.hstack((ll_to_ul,ul_to_ur,ur_to_lr,lr_to_ll))
        step = int(np.round(path.shape[1]/n_updates))
        
        sps = np.vstack((path[0][0::step], path[1][0::step]))

# agent lead/lagging
if lead_lag != 0:
    sps = np.roll(sps,-lead_lag)

# %% Run simulation 

# pre-allocating variables
gammas = np.zeros(n_updates) 
lambds = np.zeros(n_updates) 
es = np.zeros(n_updates)

x_change = np.zeros(n_updates)
y_change = np.zeros(n_updates)
p_pos = np.zeros((2,n_updates))
Ps = np.zeros(n_updates) 
Is = np.zeros(n_updates) 
Ds = np.zeros(n_updates)

# initial position and velocity for pursuer
vel_ratio = 0.7 # <1 means slower than target, >1 means faster

# match velocity to that of average of target
v_p = np.mean(get_velocity(sps))*vel_ratio 
p_pos[:,0] = [0,dim_y/2+50] # X,Y pos

# PID settings
Kp = 1
Ki = 0
Kd = 0
win_size = 100

# simulate
for i in range(n_updates):

    # Current part
    range_vec = sps[:,i] - p_pos[:,i]
    lambd = get_angle_atan2(range_vec, ref_vec)

    # unwrapping angle (to avoid error whipping)
    if i>1:
        rads = [lambds[i-1], lambd]
        rads_unwraped = np.unwrap(rads, period=np.pi)
        lambd = rads_unwraped[1]

    # avoid crashing due to i+1
    if i < (n_updates-1): 

        # Update controller
        es[i] = lambd-gammas[i]
        gammas[i+1], P,I,D = pid(i,es,gammas,Kp=Kp,Ki=Ki,Kd=Kd,int_win_size=win_size)

        # Update Process (kinematics)
        x_change[i] = np.cos(gammas[i+1])*v_p
        y_change[i] = np.sin(gammas[i+1])*v_p

        p_pos[0,i+1] = p_pos[0,i] + x_change[i]
        p_pos[1,i+1] = p_pos[1,i] + y_change[i]

    # store variables
    lambds[i] = lambd
    Ps[i] = P
    Is[i] = I
    Ds[i] = D

# Gridpsec
fig = plt.figure(figsize=(12, 8), layout="constrained")
spec = fig.add_gridspec(3, 4)
ax_traj = fig.add_subplot(spec[0:2, 0:2])
ax_angle = fig.add_subplot(spec[0:1, 2:])
ax_error = fig.add_subplot(spec[1:2, 2:])
ax_control = fig.add_subplot(spec[2:3, 2:])
ax_dis = fig.add_subplot(spec[2:, 0:2])

ax_traj.scatter(sps[0,:], sps[1,:], c = 'r', s = 5)
ax_traj.scatter(p_pos[0,:], p_pos[1,:], c = 'g', s = 5)
ax_traj.set_xlim([0, dim_x])
ax_traj.set_ylim([0, dim_y])

pos_diffs = (sps-p_pos)**2
dis = np.sqrt(pos_diffs[0,:]+pos_diffs[1,:])
ax_dis.plot(dis, label='ss dis = '+str(np.round(dis[-10],2)))
ax_dis.set_xlabel('Time')
ax_dis.set_ylabel('Distance (a.u.)')
ax_dis.set_ylim([0,max(dis)])
ax_dis.legend(loc='best',frameon=False)

ax_angle.plot(ts,np.rad2deg(wrap_angles(lambds)),'g-',label='lambda (SP)')
ax_angle.plot(ts,np.rad2deg(wrap_angles(gammas)),'b-',label='gamma (PV)')
ax_angle.set_ylabel('Angles (º)')    
ax_angle.legend(loc='best', frameon=False)
ax_angle.set_title('Params: ' + 'Kp='+str(Kp) +' Ki='+str(Ki) +
          ' int_win_size='+str(win_size) +' Kd='+str(Kd) + '\n Lead(+)/Lag(-): ' + str(lead_lag))

ax_error.axhline(0,c = 'r')
ax_error.plot(ts,es,'g-', label='ss angle error = '+str(np.round(es[-10],3)))
ax_error.set_ylabel('Error')    
ax_error.legend(loc='best', frameon=False)

ax_control.plot(ts,Ps,label='Prop')
ax_control.plot(ts,Is,label='Int')
ax_control.plot(ts,Ds,label='Der')
ax_control.set_ylabel("Controllers' effort")    
ax_control.set_xlabel('Time (sec)')
ax_control.legend(loc='best',frameon=False)

params =    'Kp='+str(Kp) +'_Ki='+str(Ki) +'_int_win_size='+str(win_size) \
            +'_Kd='+str(Kd) + '_delay='+str(lead_lag) + '_velratio='+str(vel_ratio) + '.png'

if exp == True:
    path = fd_path + 'real_flies/va+vs_cont/' + data_path[:-4] + '/' # -4 removes ".npy"
elif exp == False:
    path = fd_path + 'playground_figs/va+vs_cont/' + traj + '/'

os.makedirs(path, exist_ok=True)
fig.savefig(path + params)
# %%
