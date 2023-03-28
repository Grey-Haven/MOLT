import numpy as np
import numba as nb
from advance import *


def shuffle_steps(u):
    """
    Performs the data swap required prior to advancing to the next time step.
    
    This performs the required data transfers for the multistep methods, which store
    a total of several levels in time. The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    """
    
    # Identify the number of time levels
    num_levels = u.shape[0]
    
    # Transfer the time history starting from the oldest available data
    for level in range(num_levels-1):
        
        u[level] = u[level + 1]
    
    return None

@nb.njit([nb.void(nb.float64[:,:])], cache=False, boundscheck=False)
def shuffle_2D_boundary_data(u_bdry):
    """
    Performs the data swap for the time history along the boundaries of
    a 2D grid stored with the ordering convention (space_idx, time_idx)
    
    Compared to "shuffle_steps", this function is designed to work with
    the boundary data, where the time levels are stores along columns, rather
    than rows. This choice was made because the solvers make frequent use of 
    the boundary history in the outflow updates.
    """
    
    # Identify the number of time levels (stored in columns...)
    N_grid = u_bdry.shape[0]
    num_levels = u_bdry.shape[1]
    
    # Transfer the time history starting from the oldest available data
    for idx in range(N_grid):
        for level in range(num_levels-1):
            u_bdry[idx, level] = u_bdry[idx, level + 1]
    
    return None


#-------------------------------------------------------------
# BDF field advances
#-------------------------------------------------------------

#
# This is a specialized method for the 2D heating test
#
def BDF1_combined_per_advance(u, dudx, dudy, src_data, 
                              x, y, t_n, dx, dy, dt, c, beta_BDF):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF1_ddx_advance_per(dudx, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_ddy_advance_per(dudy, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    return None


def BDF2_combined_per_advance(u, dudx, dudy, src_data, 
                              x, y, t_n, dx, dy, dt, c, beta_BDF):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF2_ddx_advance_per(dudx, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_ddy_advance_per(dudy, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    return None


#
# This is a specialized method for the expanding beam problem
#
def BDF1_combined_dir_advance(u, dudx, dudy, src_data, 
                              x, y, t_n, dx, dy, dt, c, beta_BDF):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF1_ddx_advance_dir(dudx, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_ddy_advance_dir(dudy, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_advance_dir(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    return None


def BDF2_combined_dir_advance(u, dudx, dudy, src_data, 
                              x, y, t_n, dx, dy, dt, c, beta_BDF):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF2_ddx_advance_dir(dudx, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_ddy_advance_dir(dudy, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_advance_dir(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    return None



def BDF2_combined_out_advance(u, dudx, dudy, src_data,
                              A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                              bdry_hist_ax, bdry_hist_bx, 
                              bdry_hist_ay, bdry_hist_by,
                              x, y, t_n, dx, dy, dt, c, beta_BDF):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    # We don't need to update the history of A and B along the lines.
    BDF2_ddy_advance_out(dudy, u, src_data, x, y, t_n,
                         A_nm1_x.copy(), B_nm1_x.copy(), 
                         A_nm1_y.copy(), B_nm1_y.copy(),
                         bdry_hist_ax.copy(), bdry_hist_bx.copy(), 
                         bdry_hist_ay.copy(), bdry_hist_by.copy(),
                         dx, dy, dt, c, beta_BDF)
    
    # We don't need to update the history of A and B along the lines.
    BDF2_ddx_advance_out(dudx, u, src_data, x, y, t_n,
                         A_nm1_x.copy(), B_nm1_x.copy(), 
                         A_nm1_y.copy(), B_nm1_y.copy(),
                         bdry_hist_ax.copy(), bdry_hist_bx.copy(), 
                         bdry_hist_ay.copy(), bdry_hist_by.copy(),
                         dx, dy, dt, c, beta_BDF)
    
    # Can safely update the boundary history
    BDF2_advance_out(u, src_data, x, y, t_n,
                     A_nm1_x, B_nm1_x, 
                     A_nm1_y, B_nm1_y,
                     bdry_hist_ax, bdry_hist_bx, 
                     bdry_hist_ay, bdry_hist_by,
                     dx, dy, dt, c, beta_BDF)
    
    return None

#-------------------------------------------------------------
# Specialized methods for parabolic equations
#-------------------------------------------------------------

def BDF2_combined_per_advance_parabolic(u, dudx, dudy, src_data, 
                                        x, y, t_n, dx, dy, dt, alpha):
    """
    Performs the derivative and field advance function for a 2-D scalar field
    modeled as a parabolic equation.
    
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF2_ddx_advance_per_parabolic(dudx, u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    BDF2_ddy_advance_per_parabolic(dudy, u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    BDF2_advance_per_parabolic(u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    return None

def BDF2_combined_dir_advance_parabolic(u, dudx, dudy, src_data, 
                                        x, y, t_n, dx, dy, dt, alpha):
    """
    Performs the derivative and field advance function for a 2-D scalar field
    modeled as a parabolic equation.
    
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    """
    
    BDF2_ddx_advance_dir_parabolic(dudx, u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    BDF2_ddy_advance_dir_parabolic(dudy, u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    BDF2_advance_dir_parabolic(u, src_data, x, y, t_n, dx, dy, dt, alpha)
    
    return None

#-------------------------------------------------------------
# Centered advances (central function and BDF derivative)
#-------------------------------------------------------------

def mixed2_combined_per_advance(u, dudx, dudy, src_data_old, src_data_new,
                                x, y, t_n, dx, dy, dt, c, beta_BDF, beta_C):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    
    Note: BDF requires the FULL time history while the central scheme only uses
    a total of 3 time levels.
    """
    
    BDF2_ddx_advance_per(dudx, u, src_data_new, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_ddy_advance_per(dudy, u, src_data_new, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    central2_advance_per(u[-3:,:,:], src_data_old, x, y, t_n, dx, dy, dt, c, beta_C)
    
    return None



def mixed2_combined_dir_advance(u, dudx, dudy, src_data_old, src_data_new,
                                x, y, t_n, dx, dy, dt, c, beta_BDF, beta_C):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    
    Note: BDF requires the FULL time history while the central scheme only uses
    a total of 3 time levels.
    """
    
    BDF2_ddx_advance_dir(dudx, u, src_data_new, x, y, 
                         t_n, dx, dy, dt, c, beta_BDF)
    
    BDF2_ddy_advance_dir(dudy, u, src_data_new, x, y, 
                         t_n, dx, dy, dt, c, beta_BDF)
    
    central2_advance_dir(u[-3:,:,:], src_data_old, x, y, 
                         t_n, dx, dy, dt, c, beta_C)
    
    return None


def mixed2_combined_out_advance(u, dudx, dudy, src_data_old, src_data_new,
                                A_nm1_x_BDF, B_nm1_x_BDF, A_nm1_y_BDF, B_nm1_y_BDF,
                                A_nm1_x_C, B_nm1_x_C, A_nm1_y_C, B_nm1_y_C,
                                bdry_hist_ax_BDF, bdry_hist_bx_BDF, 
                                bdry_hist_ay_BDF, bdry_hist_by_BDF,
                                bdry_hist_ax_C, bdry_hist_bx_C, 
                                bdry_hist_ay_C, bdry_hist_by_C,
                                x, y, t_n, dx, dy, dt, c, beta_BDF, beta_C):
    """
    Performs the derivative and field advance function for a 2-D scalar field.
    The function assumes we are working with a scalar field,
    but it can be called on the scalar components of a vector field.
    
    Shuffles for time stepping are performed later, outside of this utility.
    
    Note: BDF requires the FULL time history while the central scheme only uses
    a total of 3 time levels. The BDF-2 scheme requires  a total of 4 time levels.
    """
    
    # We don't need to update the history of A and B along the lines.
    BDF2_ddx_advance_out(dudx, u, src_data_new, x, y, t_n, 
                         A_nm1_x_BDF.copy(), B_nm1_x_BDF.copy(), 
                         A_nm1_y_BDF.copy(), B_nm1_y_BDF.copy(),
                         bdry_hist_ax_BDF.copy(), bdry_hist_bx_BDF.copy(), 
                         bdry_hist_ay_BDF.copy(), bdry_hist_by_BDF.copy(),
                         dx, dy, dt, c, beta_BDF)
    
    # We can now update the history of A and B along the lines.
    BDF2_ddy_advance_out(dudy, u, src_data_new, x, y, t_n, 
                         A_nm1_x_BDF, B_nm1_x_BDF, 
                         A_nm1_y_BDF, B_nm1_y_BDF,
                         bdry_hist_ax_BDF, bdry_hist_bx_BDF,
                         bdry_hist_ay_BDF, bdry_hist_by_BDF,
                         dx, dy, dt, c, beta_BDF)
    
    # Update the history of A and B along the lines for the central scheme (no copy required)
    central2_advance_out(u[-3:,:,:], src_data_old, x, y, t_n, 
                         A_nm1_x_C, B_nm1_x_C, 
                         A_nm1_y_C, B_nm1_y_C,
                         bdry_hist_ax_C, bdry_hist_bx_C,
                         bdry_hist_ay_C, bdry_hist_by_C,
                         dx, dy, dt, c, beta_C)
    
    return None














