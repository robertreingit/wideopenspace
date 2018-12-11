# -*- encoding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle 
from scipy import signal


def calculate_velocity(player, freq = 25.0):
    """Calculates the velocity of a player in x and y coordinates.
    
        Args:
            player = A pandas dataframe containing the x and y coordinates.
            freq = Recording frequency in Hz {default: 25.0}.
        Returns:
            A pandas dataframe containing vx and vy.
    """
    delta = player.diff()
    return (delta[:-1] + delta[1:]) * freq / 2.0


def calculate_angle_from_velocity(p_xy_v):
    """Calculates the directional angle from the velocity data.

        Args:
            p_xy_v: player x-y-velocity DataFrame
        returns:
            the angle as a DataFrame
    """
    return np.arctan2(p_xy_v.y, p_xy_v.x) 


# Calculating rotation matrix 2x2
def generate_rotation_2x2_matrix(theta):
    """Generate a 2x2 rotation matrix based on the angle theta

        Bornn formula No. (16)
        
        Args:
            theta: Angle between pitch baseline and player velocity direction
        Returns:
            a 2x2 rotation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    mat = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]])
    mat[abs(mat) < np.finfo('float64').eps] = 0.0
    return mat


# Calculating direction matrix
def generate_direction_2x2_matrix(s_x, s_y):
    """Calculates the direction matrix based on the player velocities.
    
        Born formulate No. (17)
        
        Args:
            s_x = player velocity in x-direction
            s_y = player velocity in y-direction
        Returns:
            a 2x2 diagonal matrix.
    """
    return np.diag((s_x,s_y))


# Bornn function No. (18)
def calculate_velocity_magnitude(p_vel):
    """Calculates the velocity magnitude for a player.
    
        Args:
            p_vel: numpy array nx2 of velocities (x,y)
        Returns:
            a vector of the resulting velocity
    """
    assert(p_vel.shape[1] == 2)
    return np.sqrt(np.sum(p_vel**2, axis=1))

def calculate_S_rat_i(p_vel):
    """Calculates the normalized velocity.
    
        Args:
            p_vel: numpy array nx2 of velocities (x,y)
        Returns:
            the normalization value.
    """
    s = calculate_velocity_magnitude(p_vel)
    return s**2 / 13.0**2

def z_butter_filter(sig, cutoff, freq, order = 2):
    """Low-pass zero-phase butterworth filter with cut-off freq.

        Args:
            sig:    numpy array to be filtered.
            cutoff: cut-off frequency
            freq:   recording frequency
            order:  order of filter
        Returns:
            the filtered signal.
    """
    b, a = signal.butter(order, 2.0 * cutoff / freq)
    return signal.filtfilt(b, a, sig, axis = 0)


def filter_velocity(df, freq = 3.0):
    """
    """
    no_rows = df.shape[0]
    yf = np.full((no_rows,2), np.nan)
    yf[1:-1,:] = z_butter_filter(df[['x','y']].dropna(), freq, 25.0)
    return pd.DataFrame(yf, columns=['x','y'])

def plot_pitch():
    rect = Rectangle((-105.0/2.0, -68/2.0), 105.0, 68, facecolor='green',
            edgecolor = 'white',
            linewidth = 2)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    #dat = pd.read_csv('test_data.csv')
    p1 = dat.iloc[:,:2]
    ball = dat[['ball_x','ball_y']]
    p1.columns = ['x','y']
    plt.clf()
    #plot_pitch()
    #plt.plot(p1.x, p1.y, 'y-')
    p1_v = calculate_velocity(p1)
    #plt.plot(p1_v)
    ## filter velocity
    p1_v_f = filter_velocity(p1_v, 2.)
    #plt.plot(p1_v_f,'r--')
    ## calculate the directional angle
    angle = calculate_angle_from_velocity(p1_v_f)
    ## calculate the velocity magnitude
    mag = calculate_velocity_magnitude(p1_v_f)
    #plt.plot(angle)
    R = generate_rotation_2x2_matrix(angle[1])
    S = generate_direction_2x2_matrix(*p1_v.iloc[1,:])
    S_rat = calculate_S_rat_i(p1_v_f)
    ## Distance to ball
    ## Transfer function
    ## Generate S_i_t
    ## Calculate covariance
    ## calculate mu

