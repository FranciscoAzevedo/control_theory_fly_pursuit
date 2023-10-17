"""
    Rotation work for Eugenia Chiappe May-June 2023
    Author: Francisco Moreira de Azevedo, 2023
    e-mail: francisco.azevedo@research.fchampalimaud.org

    Implementation of parallel Va and Vs controllers
"""

# %% Imports and plot params
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import imageio
import os
import util
%matplotlib qt

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.linewidth': 1,
         'figure.dpi': 120,
         }

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams.update(params)

fd_path = '/home/paco-laptop/Desktop/paco/code/control_theory_fly_pursuit/'

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
    # start accumulating error even if smaller than int win size
    elif (iter < int_win_size) & (iter >= 1):
        ie = np.mean(es[:iter])
    # rolling window of error
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

# lead_lag<0 means delayed, >0 means looking ahead
lead_lag_va = 0
lead_lag_vs = 0

# coordinate referential
ref_vec = np.array([0.01,0]) # important that first value is small

# Defines whether you load an experimental path or pre-programmed
exp = True
data_path = fd_path + 'real_flies/data/parallel/sps_prof1.npy' # created from explore_real_data.py
fly_pos_path = fd_path + 'real_flies/data/parallel/fly_pos_prof1.npy'

session_limit = 500 # use only portion of a exp session

if exp == True:
    data = np.load(data_path)
    data = data[:,0:session_limit]
    sps = data[0:2,:]

    fly_pos_data = np.load(fly_pos_path)
    fly_pos = fly_pos_data[0:2,0:session_limit]

    # simulation config
    t = data[2,-1] - data[2,0] # full duration
    n_updates = len(data[2,:])
    ts = np.linspace(0,t,n_updates)
    frame_len = ts[1]

    # playground dimensions
    dim_x = max(abs(sps[0,:]))*2
    dim_y = max(abs(sps[1,:]))*2

    sps = util.nan_cleaner_and_shift(sps, dim_x, dim_y)
    fly_pos = util.nan_cleaner_and_shift(fly_pos, dim_x, dim_y)

elif exp == False:
    # simulation config
    t = 10 # seconds
    n_updates = 200
    ts = np.linspace(0,t,n_updates)
    frame_len = ts[1]

    # playground dimensions
    dim_x = 400
    dim_y = 300

    # pre-programmed paths (must begin and end at same point)

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
        sps[0,:] = np.linspace(20,dim_x, num=n_updates)  
        sps[1,:] = dim_y/2

    # line with jitter noise
    if traj == 'jitter_line':
        sps[0,:] = np.linspace(0,dim_x, num=n_updates)  

        # jitter
        noise_sigma = 10*frame_len # 10 mm/s to mm/frame_len
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
if lead_lag_va != 0:
    sps_va = np.roll(sps,-lead_lag_va)
    sps_va = util.lead_lag_wrapping(sps_va, lead_lag_va)
else:
    sps_va = sps

if lead_lag_vs != 0:
    sps_vs = np.roll(sps,-lead_lag_vs)
    sps_vs = util.lead_lag_wrapping(sps_vs, lead_lag_vs)
else:
    sps_vs = sps

# %% Run simulation 

# animation
animate = False

# pre-allocating variables
gammas = np.zeros(n_updates) # controlled in Va
lambds = np.zeros(n_updates)
es_va = np.zeros(n_updates) # same as es_vs if there is no lag difference

lambds_vs = np.zeros(n_updates) # controlled in Vs
e_primes = np.zeros(n_updates) 
es_vs = np.zeros(n_updates) # same as es_va if there is no lag difference

mag_vs = np.zeros(n_updates)
zeros = np.zeros(n_updates) # since sideways velocity does not accumulate like angular

x_change = np.zeros(n_updates)
y_change = np.zeros(n_updates)
p_pos = np.zeros((2,n_updates))
P_va = np.zeros(n_updates) 
I_va = np.zeros(n_updates) 
D_va = np.zeros(n_updates)

P_vs = np.zeros(n_updates) 
I_vs = np.zeros(n_updates) 

# Velocity for pursuer
vel_ratio = 0.9 # <1 means slower than target, >1 means faster

t_vel = util.get_velocity(sps) # target velocities
t_vel[-1] = t_vel[-2]

v_p = t_vel*vel_ratio # match velocity to that of real time of target
p_pos[:,0] = [dim_x,0] # initial position X,Y

# Biological limits to the movement
vs_max = 10 * frame_len # 10 mm/s to mm/frame
va_max = np.deg2rad(1000)*frame_len # 1000 degrees/s to rads/frame

# general PID
win_size = 20

# PID settings for v_a
va_ON = True
if va_ON == False:
    Kp_va = 0
    Ki_va = 0
    Kd_va = 0
elif va_ON == True:
    Kp_va = 1
    Ki_va = 0
    Kd_va = 0.5

# PID settings for v_s
vs_ON = True
if vs_ON == False:
    Kp_vs = 0
    Ki_vs = 0
    Kd_vs = 0
elif vs_ON == True:
    Kp_vs = 1
    Ki_vs = 0.5
    Kd_vs = 0

if animate == True:
    fig,axes = plt.subplots()
    plt.ion()

    images = [] # for gif

# simulate
for i in range(n_updates):

    # Current part - Va
    range_vec_unnorm = sps_va[:,i] - p_pos[:,i]
    lambd = util.get_angle_atan2(range_vec_unnorm, ref_vec)

    # unwrapping angle (to avoid error whipping)
    if i>1:
        rads = [lambds[i-1], lambd]
        rads_unwraped = np.unwrap(rads, period=np.pi)
        lambd = rads_unwraped[1]

    # Current part - Vs
    range_vec_unnorm = sps_vs[:,i] - p_pos[:,i]
    lambd_vs = util.get_angle_atan2(range_vec_unnorm, ref_vec)

    # unwrapping angle (to avoid error whipping)
    if i>1:
        rads = [lambds_vs[i-1], lambd_vs]
        rads_unwraped = np.unwrap(rads, period=np.pi)
        lambd_vs = rads_unwraped[1]

    # avoid crashing due to i+1 in last iteration
    if i < (n_updates-1): 

        # Update Va controller
        es_va[i] = lambd-gammas[i]
        gammas[i+1], P,I,D = pid(i,es_va,gammas,Kp=Kp_va,Ki=Ki_va,Kd=Kd_va,int_win_size=win_size)

        # clipping max va
        delta_gamma = gammas[i+1] - gammas[i]
        if delta_gamma > va_max:
            gammas[i+1] = gammas[i] + va_max
        elif delta_gamma < -va_max:
            gammas[i+1] = gammas[i] - va_max

        # Update Vs controller
        es_vs[i] = lambd_vs-gammas[i]
        e_primes[i+1], P_v, I_v, _ = pid(i,es_vs,zeros,Kp=Kp_vs,Ki=Ki_vs,Kd=Kd_vs,int_win_size=win_size)
        d = np.linalg.norm(range_vec_unnorm)

        # derived from ego equations
        mag_vs[i] = d*(np.sin(es_vs[i]) - np.cos(es_vs[i])*np.tan(es_vs[i]-e_primes[i+1]))

        # clipping max vs
        if mag_vs[i] > vs_max:
            mag_vs[i] = vs_max
        elif mag_vs[i] < -vs_max:
            mag_vs[i] = -vs_max

        # Update Process (kinematics) inputting va and vs
        x_change[i] = np.cos(gammas[i+1])*v_p[i] + mag_vs[i]*np.cos(gammas[i]+np.pi/2)
        y_change[i] = np.sin(gammas[i+1])*v_p[i] + mag_vs[i]*np.sin(gammas[i]+np.pi/2)
        
        # need to normalize velocities
        if va_ON == True and vs_ON == True:
            rho, phi = util.cart2pol(x_change[i], y_change[i]) 
            rho = v_p[i] # forcing it to not exceed stipulated magnitude of velocity

            x_change[i], y_change[i] = util.pol2cart(rho, phi)

        p_pos[0,i+1] = p_pos[0,i] + x_change[i]
        p_pos[1,i+1] = p_pos[1,i] + y_change[i]

    # store variables
    lambds[i] = lambd
    lambds_vs[i] = lambd_vs
    P_va[i] = P
    I_va[i] = I
    D_va[i] = D
    P_vs[i] = P_v
    I_vs[i] = I_v

    # Animation stuff
    if animate == True:
        
        axes.scatter(p_pos[0,i], p_pos[1,i], c='xkcd:lime green', s = 6) # pursuer
        ori = [p_pos[0,i]+np.cos(gammas[i])*10, p_pos[1,i]+np.sin(gammas[i])*10]
        axes.plot([p_pos[0,i], ori[0]], [p_pos[1,i], ori[1]], c='white') # pursuers' ori

        axes.scatter(sps[0,i], sps[1,i], c = 'xkcd:purple pink', s = 6) # target

        if exp == True:
            axes.scatter(fly_pos[0,i], fly_pos[1,i], c='xkcd:light cyan', s = 6) # real fly

        axes.set_xlim([-20, dim_x+20])
        axes.set_ylim([-20, dim_y+20])
        axes.patch.set_facecolor('k')
        fig.patch.set_facecolor('k')
        axes.axis('off')
        plt.pause(0.01)
        fig.canvas.draw()

        filename= fd_path+'gif_images/'+str(1000+i)+'.png'
        plt.savefig(filename)
        images.append(imageio.imread(filename))
        axes.clear()

if animate == True:
    imageio.mimsave(fd_path + 'playground_figs/' + 'pursuit.gif', images)

# Trajectories plot
fig, axes = plt.subplots(figsize = (4,4))
axes.scatter(sps[0,:], sps[1,:], c = 'r', s = 2, label = 'target')
axes.scatter(p_pos[0,:], p_pos[1,:], c = 'g', s = 2, label = 'pursuer')
axes.set_xlim([0, dim_x+5])
axes.set_ylim([0, dim_y+5])
axes.axis('off')
axes.legend(frameon = False)

# %% Full plot

# Gridpsec
fig = plt.figure(figsize=(12, 10), layout="constrained")
spec = fig.add_gridspec(3, 6)
ax_traj = fig.add_subplot(spec[0:2, 0:2])
ax_angle = fig.add_subplot(spec[0:1, 2:4])
ax_error = fig.add_subplot(spec[1:2, 2:4])
ax_control = fig.add_subplot(spec[2:3, 2:4])
ax_dis = fig.add_subplot(spec[2:, 0:2])

# Trajectory and distance
ax_traj.scatter(sps[0,:], sps[1,:], c = 'r', s = 2)
ax_traj.scatter(p_pos[0,:], p_pos[1,:], c = 'g', s = 2)
ax_traj.set_xlim([-5, dim_x+5])
ax_traj.set_ylim([-5, dim_y+5])

pos_diffs = (sps-p_pos)**2
dis = np.sqrt(pos_diffs[0,:]+pos_diffs[1,:])
ax_dis.plot(ts,dis, label='ss dis = '+str(np.round(dis[-10],2)))
ax_dis.set_xlabel('Time')
ax_dis.set_ylabel('Distance (a.u.)')
ax_dis.set_ylim([0,max(dis)])
ax_dis.legend(loc='best',frameon=False)

# Va controller
v_a = np.roll(gammas,-1)-gammas
v_a = np.rad2deg(v_a)/frame_len # convert to degree/sec to plot

ax_angle.plot(ts[:-1],v_a[:-1],'xkcd:bright blue')
ax_angle.set_ylabel('Va (°/s)')    
ax_angle.set_title('Params: ' + 'Kp='+str(Kp_va) +' Ki='+str(Ki_va) +
          ' int = '+str(round(win_size*frame_len*1000)) +'ms Kd='+str(Kd_va))
ax_angle.set_xticks([], [])

ax_error.axhline(0,c = 'r')
ax_error.plot(ts,es_va,'xkcd:green', label='ss angle error = '+str(np.round(es_va[-10],3)))
ax_error.set_ylabel('Error')    
ax_error.legend(loc='best', frameon=False)
ax_error.set_xticks([], [])

ax_control.plot(ts,P_va,label='Prop')
ax_control.plot(ts,I_va,label='Int')
ax_control.plot(ts,D_va,label='Der')
ax_control.set_ylabel("Controllers' effort")    
ax_control.set_xlabel('Time (sec)')
ax_control.legend(loc='best',frameon=False)

# Vs controller
ax_vs = fig.add_subplot(spec[0:1, 4:])
ax_error_vs = fig.add_subplot(spec[1:2, 4:])
ax_control_vs = fig.add_subplot(spec[2:3, 4:])

ax_vs.plot(ts,mag_vs*(1/frame_len),'xkcd:bright blue')
ax_vs.set_ylabel('Vs (mm/s)')
ax_vs.set_ylim([-10.5, 10.5])     
ax_vs.set_title('Params: ' + 'Kp='+str(Kp_vs) +' Ki='+str(Ki_vs) +
          ' int = '+str(round(win_size*frame_len*1000)) +' ms')
ax_vs.set_xticks([], [])

ax_error_vs.plot(ts, es_vs, 'xkcd:green', label='ss angle error = '+str(np.round(es_vs[-10],3)))
ax_error_vs.set_ylabel('Error')    
ax_error_vs.legend(loc='best', frameon=False)
ax_error_vs.set_xticks([], [])

ax_control_vs.plot(ts,P_vs,label='Prop')
ax_control_vs.plot(ts,I_vs,label='Int')  
ax_control_vs.set_xlabel('Time (sec)')
ax_control_vs.legend(loc='best',frameon=False)

params =    'Kp='+str(Kp_va) +'_Ki='+str(Ki_va) +'_int_win_size='+str(win_size) \
            +'_Kd='+str(Kd_va) + '_velratio='+str(vel_ratio) + '.png'

if exp == True:
    path = fd_path + 'real_flies/va+vs_cont/parallel/'
elif exp == False:
    path = fd_path + 'playground_figs/va+vs_cont/' + traj + '/'

os.makedirs(path, exist_ok=True)
fig.savefig(path + params)

# %%
