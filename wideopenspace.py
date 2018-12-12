# -*- encoding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle 
from scipy import signal
from scipy.stats import multivariate_normal


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

def calculate_S_rat_i(p_vel, max_vel = 13.0):
    """Calculates the normalized velocity.
    
        Args:
            p_vel: numpy array nx2 of velocities (x,y)
            max_vel: maximum velocity to use normalize
        Returns:
            the normalization value [0,1].
    """
    s = calculate_velocity_magnitude(p_vel)
    return s**2 / max_vel**2

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

def plot_pitch(ax = plt.gca()):
    rect = Rectangle((-105.0/2.0, -68/2.0), 105.0, 68, facecolor='green',
            edgecolor = 'white',
            linewidth = 2,
            zorder = -10)
    ax.add_patch(rect)
    plt.show()


def wideopen_transfer_func(x, min_distance = 4.0, 
                           max_distance = 10.0, cut_off = 18.0):
    """Approximate transfer function from Bornn Figure 9.
    
        Calculate the player's pitch control surface radius.
    
        Args:
            x: numpy array of x-values or scalar
            min_distance: minimun distance of player's pitch control radius
            max_distance: maximum distance of player's pitch control radius
            cut_off: distance where max_distance is reached
        Returns:
            player's control surface radius
    """

    alpha = np.log(max_distance - min_distance + 1.0 )/cut_off
    if np.isscalar(x):
        if x < cut_off:
            y = min_distance - 1.0 + np.exp(alpha * x)
        else:
            y = max_distance
    else:
        idx = x < cut_off
        y = np.full_like(x, 0.0)
        y[np.invert(idx)] = max_distance
        y[idx] = min_distance - 1.0 + np.exp(alpha*x[idx])
    return y


def distance_player_2_ball(player, ball):
    """Calculates the distance between the ball and the player.
    
        Args:
            player: player DataFrame containing x and y values.
            ball: ball DataFrame containing x and y values
        Returns:
            a pd.Series object.
    """
    return np.sqrt(np.sum((ball - player)**2, axis=1))
    
def plot_player_influence(px, py, pz, mu_i_t, p1_v_f):
    """
    """
    plt.contourf(px, py, pz, alpha=.8)
    plt.plot(mu_i_t.x, mu_i_t.y, 'ro', markersize=2)
    plt.arrow(mu_i_t.x, mu_i_t.y, p1_v_f.x, p1_v_f.y, color='r')


def calculate_influence(pos_xy, vel_xy, angle, D_i_t, grid_size = 0.5):
    """Calculates the players influence according to Fernandez & Born.

        Args:
        Returns:
    """
    # Eq. 16
    R_theta = generate_rotation_2x2_matrix(angle)
    ## Transfer function
    R_i_t =  wideopen_transfer_func(D_i_t)
    # Eq. 18
    S_rat_i = np.sqrt(np.sum(vel_xy**2))/13.0**2
    #print('{0:.2f} - {1:.2f} - {2:.2f} - {3:.2f}'.format(R_i_t, S_rat_i, vel_xy.x, vel_xy.y))
    # Eq. 19
    S_i_t = np.diag([R_i_t + R_i_t * S_rat_i, R_i_t - R_i_t * S_rat_i])
    # Eq. 20
    cov = R_theta.dot(S_i_t).dot(S_i_t).dot(R_theta.transpose())
    # Eq. 21 
    mu_i_t = pos_xy + vel_xy * 0.5

    # determine necessary grid coordinates
    grid_min = np.floor(mu_i_t - R_i_t)
    grid_max = np.ceil(mu_i_t + R_i_t)
    px, py = np.mgrid[slice(grid_min.x, grid_max.x, grid_size),
            slice(grid_min.y, grid_max.y, grid_size)]

    # generate structure for pdf-calculation.
    pos = np.empty(px.shape + (2,))
    pos[:, :, 0] = px; pos[:, :, 1] = py
    rv = multivariate_normal(mu_i_t, cov)
    pz = rv.pdf(pos)

    # zeroing out all points which are too far away from the player
    # according to ball distance
    radius_idx = np.sqrt((px - mu_i_t.x)**2 + (py - mu_i_t.y)**2) > R_i_t
    pz[radius_idx] = 0
    # normalize influence
    pz = pz / np.sum(pz)
    return px, py, pz, mu_i_t 


if __name__ == '__main__':
    dat = pd.read_csv('test_data.csv')
    ball = dat[['ball_x','ball_y']]
    ball.columns = ['x','y']
    plt.clf()
    plot_pitch(plt.gca())
    pitch_x, pitch_y = np.mgrid[-55:55:0.5, -40:40:0.5]
    pitch_grid = np.zeros(pitch_x.shape)
    frame = 90
    for i in [0,2,4,6,8,10,12,14,16,18,20]:
        player = dat.iloc[:,slice(i,i+2)]
        player.columns = ['x','y']
        player_v = calculate_velocity(player)
        player_v_f = filter_velocity(player_v, 2.)
        angle = calculate_angle_from_velocity(player_v_f)
        D_i_t = distance_player_2_ball(player, ball)

        px, py, pz, mu_i_t = calculate_influence(player.iloc[frame],
                player_v_f.iloc[frame], angle[frame], D_i_t[frame])
    
        plot_player_influence(px, py, pz, mu_i_t, player_v_f.iloc[frame])
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                idx = np.where(np.logical_and(
                    pitch_x == px[i,j],
                    pitch_y == py[i,j]))
                pitch_grid[idx] = pitch_grid[idx] + pz[i,j]

    plt.plot(ball.x[frame], ball.y[frame], 'ro', markersize=5,
            markeredgecolor='k')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
