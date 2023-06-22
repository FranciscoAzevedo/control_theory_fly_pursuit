"""
    Rotation work for Eugenia Chiappe May-June 2023
    Author: Francisco Moreira de Azevedo, 2023
    e-mail: francisco.azevedo@research.fchampalimaud.org
"""

# %% Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib qt
import imageio

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

fd_path = '/home/paco-laptop/Desktop/paco/code/control_theory_fly_pursuit/real_flies/'

# %% Load data - choose the one with highest perc of chase
dataset_type = 'parallel'
# TODO choose any arbitrary profile number

max = 0
for name in os.listdir(fd_path + 'data/' + dataset_type + '/profile1/'):
    one_table = pd.read_csv(fd_path + 'data/'+ dataset_type +'/profile1/' + name)
    chase = len(one_table[one_table['Chase'] == 1])/len(one_table)
    
    if chase > max:
        max = chase
        table_name = name

# %% Animate fly with most chase percentage
data = pd.read_csv(fd_path + 'data/'+ dataset_type +'/profile1/' + table_name)

fig,axes = plt.subplots()

plt.ion()
for idx, row in data.iterrows():

    # Animation stuff
    axes.set_xlim([-20, 20])
    axes.set_ylim([-40, 40])
    plt.scatter(row['Fly_Center_Coord_Mm_1'], row['Fly_Center_Coord_Mm_2'], c='r', s = 6)
    plt.scatter(row['Stim_Center_Coord_Mm_1'], row['Stim_Center_Coord_Mm_2'], c ='k', s = 6)
    plt.pause(0.001)

    fig.canvas.draw()
    axes.clear()

# %% Plot path and store it for simulations with agent
plt.scatter(data['Stim_Center_Coord_Mm_1'], data['Stim_Center_Coord_Mm_2'], c ='k', s = 3)
plt.scatter(data['Fly_Center_Coord_Mm_1'], data['Fly_Center_Coord_Mm_2'], c='r', s = 3)

sps = np.zeros((3, len(data)))
sps[0,:] = data.Stim_Center_Coord_Mm_1
sps[1,:] = data.Stim_Center_Coord_Mm_2
sps[2,:] = data.Time

fly_pos = np.zeros((3, len(data)))
fly_pos[0,:] = data.Fly_Center_Coord_Mm_1
fly_pos[1,:] = data.Fly_Center_Coord_Mm_2
fly_pos[2,:] = data.Time

np.save(fd_path + 'data/'+ dataset_type +'/sps_prof1', sps)
np.save(fd_path + 'data/'+ dataset_type +'/fly_pos_prof1', fly_pos)

# %%
