import numpy as np

from field_advance import *
from extension import *
from quadrature import *
from summation import *
from boundary import *

def setup_helper_fcns():
    """
    Dummy function to precompile and cache helper functions used by the MOLT schemes.
    
    Nothing is done here, so this just tests to see if they compile correctly. Correctness
    should be verified by other means.
    """
    
    # Make a dummy mesh on [0,1]^2 and apply one step
    N = 16
    dx = 1/(N-1)
    dt = 0.1
    
    x = np.linspace(0, 1.0, N, endpoint=True)
    
    t_n = 0
    kappa = 1.0
    beta = np.sqrt(2)
    
    alpha = beta/(kappa*dt)
    
    # Mesh data that includes the ghost region
    v = np.zeros([N+4])
    
    # Operators for the sweeps
    left_moving_op = np.zeros([N])
    rite_moving_op = np.zeros([N])
    
    # Extension methods
    polynomial_extension(v)
    periodic_extension(v)
    
    # Quadrature
    linear5_L(rite_moving_op, v, alpha, dx)
    linear5_R(left_moving_op, v, alpha, dx)
    
    # Quadrature (no ghost region)
    fixed3_L(rite_moving_op, v[2:-2], alpha, dx)
    fixed3_R(left_moving_op, v[2:-2], alpha, dx)
    
    # Fast summation
    fast_convolution(rite_moving_op, left_moving_op, alpha, dx)
    
    # Boundary
    A_x = 0.0
    B_x = 0.0
    
    apply_A_and_B(rite_moving_op, x, alpha, A_x, B_x)
    
    print("Success...\n")
    
    return None

def setup_BDF2_field_solvers():
    """Dummy function to precompile the field solver functions to reduce startup cost."""

    # Make a dummy mesh on [0,1]^2 and apply one step
    N = 16
    dx = 1/(N-1)
    dy = dx
    dt = 0.1
    
    x = np.linspace(0, 1.0, N, endpoint=True)
    y = np.linspace(0, 1.0, N, endpoint=True)
    
    t_n = 0
    kappa = 1.0
    beta_BDF = np.sqrt(2)
    
    # Solution and derivative data
    u = np.zeros([4,N,N])
    dudx = np.zeros([N,N])
    dudy = np.zeros([N,N])
    
    # Make a dummy array for the source
    # Needs to be an array rather than a function
    src_array = np.zeros([N,N])
    
    # Boundary history for the outflow example
    A_nm1_x = np.zeros([N])
    B_nm1_x = np.zeros([N])
    
    A_nm1_y = np.zeros([N])
    B_nm1_y = np.zeros([N])
    
    # For more higher order, need more than 3 time levels
    u_ax = np.zeros([N,3])
    u_bx = np.zeros([N,3])
    u_ay = np.zeros([N,3])
    u_by = np.zeros([N,3])

    # Two-way periodic example
    BDF2_combined_per_advance(u, dudx, dudy, src_array, 
                              x, y, t_n, dx, dy, dt, kappa, beta_BDF)
    
    print("Periodic advance succeeded\n")
    
    # Two-way Dirichlet example
    BDF2_combined_dir_advance(u, dudx, dudy, src_array, 
                              x, y, t_n, dx, dy, dt, kappa, beta_BDF)
    
    print("Dirichlet advance succeeded\n")
    
    # Two-way outflow example
    BDF2_combined_out_advance(u, dudx, dudy, src_array, 
                              A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                              u_ax, u_bx, u_ay, u_by,
                              x, y, t_n, dx, dy, dt, kappa, beta_BDF)
    
    # Functions that shuffle the time history
    shuffle_steps(u)
    shuffle_2D_boundary_data(u_ax)
    
    print("Outflow advance succeeded\n")
    
    print("Setup complete\n")
    
    return None

def setup_mixed2_field_solvers():
    """Dummy function to precompile the field solver functions to reduce startup cost."""
    
    # Make a dummy function for the source
    src_fcn = lambda x,y,t: 0.0

    # Make a dummy mesh on [0,1]^2 and apply one step
    N = 16
    dx = 1/(N-1)
    dy = dx
    dt = 0.1
    
    x = np.linspace(0, 1.0, N, endpoint=True)
    y = np.linspace(0, 1.0, N, endpoint=True)
    
    t_n = 0
    kappa = 1.0
    beta_BDF = np.sqrt(2)
    beta_C = np.sqrt(2)
    
    u = np.zeros([4,N,N])
    dudx = np.zeros([N,N])
    dudy = np.zeros([N,N])
    
    # Make a dummy array for the source
    # Needs to be an array rather than a function
    src_array = np.zeros([N,N])
    
    # In the mixed methods with outflow boundaries, we need to track the
    # boundary history for both the BDF and central schemes
    #
    # Note that in the mixed methods, we need to update the integrands
    # so we shouldn't use copies here when transferring the history data
    
    # BDF boundary history for the outflow example
    A_nm1_x_BDF = np.zeros([N])
    B_nm1_x_BDF = np.zeros([N])
    
    A_nm1_y_BDF = np.zeros([N])
    B_nm1_y_BDF = np.zeros([N])
    
    bdry_hist_ax_BDF = np.zeros([N,3])
    bdry_hist_bx_BDF = np.zeros([N,3])
    
    bdry_hist_ay_BDF = np.zeros([N,3])
    bdry_hist_by_BDF = np.zeros([N,3])
    
    # Centered boundary history for the outflow example
    A_nm1_x_C = np.zeros([N])
    B_nm1_x_C = np.zeros([N])
    
    A_nm1_y_C = np.zeros([N])
    B_nm1_y_C = np.zeros([N])
    
    bdry_hist_ax_C = np.zeros([N,3])
    bdry_hist_bx_C = np.zeros([N,3])
   
    bdry_hist_ay_C = np.zeros([N,3])
    bdry_hist_by_C = np.zeros([N,3])

    # Two-way periodic example    
    mixed2_combined_per_advance(u, dudx, dudy, src_array, src_array, 
                                x, y, t_n, dx, dy, dt, kappa, beta_BDF, beta_C)
    
    print("Periodic advance succeeded\n")
    
    # Two-way Dirichlet example
    mixed2_combined_dir_advance(u, dudx, dudy, src_array, src_array, 
                                x, y, t_n, dx, dy, dt, kappa, beta_BDF, beta_C)
    
    print("Dirichlet advance succeeded\n")
    
    # Two-way outflow example
    mixed2_combined_out_advance(u, dudx, dudy, src_array, src_array, 
                                A_nm1_x_BDF, B_nm1_x_BDF, A_nm1_y_BDF, B_nm1_y_BDF,
                                A_nm1_x_C, B_nm1_x_C, A_nm1_y_C, B_nm1_y_C,
                                bdry_hist_ax_BDF, bdry_hist_bx_BDF,
                                bdry_hist_ay_BDF, bdry_hist_by_BDF,
                                bdry_hist_ax_C, bdry_hist_bx_C,
                                bdry_hist_ay_C, bdry_hist_by_C,
                                x, y, t_n, dx, dy, dt, kappa, beta_BDF, beta_C)
    
    print("Outflow advance succeeded\n")
    
    print("Setup complete\n")
    
    return None

def setup_BDF2_parabolic_field_solvers():
    """Dummy function to precompile the field solver functions to reduce startup cost."""

    # Make a dummy mesh on [0,1]^2 and apply one step
    N = 16
    D = 1.0
    dx = 1/(N-1)
    dy = dx
    dt = 0.1
    
    x = np.linspace(0, 1.0, N, endpoint=True)
    y = np.linspace(0, 1.0, N, endpoint=True)
    
    t_n = 0
    alpha = np.sqrt(3/(2*D*dt))
    
    # Solution and derivative data
    u = np.zeros([3,N,N])
    dudx = np.zeros([N,N])
    dudy = np.zeros([N,N])
    
    # Make a dummy array for the source
    # Needs to be an array rather than a function
    src_array = np.zeros([N,N])

    # Two-way periodic example
    BDF2_combined_per_advance_parabolic(u, dudx, dudy, src_array, 
                              x, y, t_n, dx, dy, dt, alpha)
    
    print("Periodic advance succeeded\n")
    
    # Two-way Dirichlet example
    BDF2_combined_dir_advance_parabolic(u, dudx, dudy, src_array, 
                              x, y, t_n, dx, dy, dt, alpha)
    
    print("Dirichlet advance succeeded\n")
    
    print("Setup complete\n")
    
    return None






