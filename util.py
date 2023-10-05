"""
    Rotation work for Eugenia Chiappe May-June 2023
    Author: Francisco Moreira de Azevedo, 2023
    e-mail: francisco.azevedo@research.fchampalimaud.org

    Helper functions for controllers
"""

# %% Helper functions
import numpy as np
import math

# coordinate transformations

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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

# computing instantaneous velocity
def get_velocity(XY_pos):
    
    XY_pos_shift_fwd = np.roll(XY_pos,-1) # shift one array forward

    # compute difference and square
    # does both subtraction and power operations to X and Y seperately
    pos_diffs = (XY_pos_shift_fwd-XY_pos)**2 

    norms = np.sqrt(pos_diffs[0,:]+pos_diffs[1,:])
    
    return np.array(norms) # last is always wrong in this implementation

# fixes wrapping at begining not working when introducing lead/lags
def lead_lag_wrapping(sps, lead_lag):
    
    if lead_lag < 0: 
        sps[:,0:(-lead_lag)] = np.tile(sps[:,-lead_lag+1], [-lead_lag,1]).T
    elif lead_lag > 0:
        sps[:,(-lead_lag):] = np.tile(sps[:,-lead_lag-1], [lead_lag,1]).T

    return sps

# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_cleaner_and_shift(xys, dim_x, dim_y):
    """
        Cleans nans out of X,Y position vector
        Shifts coord system 
    """

    # Filling up nans with previous value registered
    nans, x = nan_helper(xys[0,:])
    xys[0,:][nans]= np.interp(x(nans), x(~nans), xys[0,:][~nans])

    nans, x = nan_helper(xys[1,:])
    xys[1,:][nans]= np.interp(x(nans), x(~nans), xys[1,:][~nans])

    # changing referential to lower left instead of center
    xys[0,:] = xys[0,:] + dim_x/2
    xys[1,:] = xys[1,:] + dim_y/2

    return xys