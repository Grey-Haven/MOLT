import numpy as np
import numba as nb
from scipy.optimize import fsolve
import warnings


#
# This file contains the utility functions for particles and the integrators
#


@nb.njit([nb.float64(nb.float64[:,:], nb.float64, nb.float64, nb.float64[:], nb.float64[:],
                     nb.float64, nb.float64)], cache = True, boundscheck=True)
def gather_2D(F_mesh, x1_p, x2_p, x, y, dx, dy):
    """
    Gathers (uniform) mesh data to a single particle with coordinates (x1_p, x2_p).
    
    This function provides the mesh-to-particle mapping using linear splines.
    """
    
    # Logical indices
    x_idx = int( np.abs(x1_p - x[0])/dx )
    y_idx = int( np.abs(x2_p - y[0])/dy )
    
    # Fraction of the cell relative to the left grid point in each direction
    f_x = (x1_p - x[x_idx])/dx
    f_y = (x2_p - y[y_idx])/dy

    # Distribute each field to the particle
    F_p = F_mesh[x_idx, y_idx]*(1 - f_x)*(1 - f_y)
    F_p += F_mesh[x_idx, y_idx+1]*(1 - f_x)*f_y
    F_p += F_mesh[x_idx+1, y_idx]*f_x*(1 - f_y)
    F_p += F_mesh[x_idx+1, y_idx+1]*f_x*f_y

    return F_p


@nb.njit([nb.void(nb.float64[:,:], nb.float64, nb.float64, nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64, nb.float64)], cache = True, boundscheck=True)
def scatter_2D(F_mesh, x1_p, x2_p, x, y, dx, dy, weight):
    """
    Scatters a single particle with coordinates (x1_p, x2_p) onto 
    uniform mesh points with an area rule.
    
    This function uses linear splines to map particle data onto a mesh.
    """
    
    # Logical indices
    x_idx = int( np.abs(x1_p - x[0])/dx )
    y_idx = int( np.abs(x2_p - y[0])/dy )
    
    # Fraction of the cell relative to the left grid point in each direction
    f_x = (x1_p - x[x_idx])/dx
    f_y = (x2_p - y[y_idx])/dy

    # Weight the particle info to the mesh
    F_mesh[x_idx, y_idx] += weight*(1 - f_x)*(1 - f_y)
    F_mesh[x_idx, y_idx+1] += weight*(1 - f_x)*f_y
    F_mesh[x_idx+1, y_idx] += weight*f_x*(1 - f_y)
    F_mesh[x_idx+1, y_idx+1] += weight*f_x*f_y

    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, 
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64, nb.float64, nb.float64, nb.float64,
                  nb.float64[:,:], nb.float64, nb.float64)], cache = True)
def particle_to_mesh_2D3P(rho_mesh, J3_mesh, A3, x, y, dx, dy,
                          x1_ions, x2_ions, p1_ions, p2_ions, p3_ions,
                          x1_elec, x2_elec, p1_elec, p2_elec, p3_elec,
                          q_ions, q_elec, r_ions, r_elec, mu_r, 
                          cell_volumes, w_ions, w_elec):
    """
    Computes the sources for the field solvers, i.e., the charge density and the current density.
    This mapping is to be used for the Bennett pinch problem with Molei Tao's method.
    """
    
    # Number of simulation particles for each species
    N_ions = x1_ions.size
    N_elec = x1_elec.size

    # Clear the contents of the mesh arrays
    rho_mesh[:,:] = 0.0
    
    #J_mesh[:,:,:] = 0.0
    J3_mesh[:,:] = 0.0
    
    # Scatter ion data to the mesh
    for i in range(N_ions):
        
        # Step 1: Ion charge density
        weight = w_ions*q_ions
        scatter_2D(rho_mesh, x1_ions[i], x2_ions[i], x, y, dx, dy, weight)

        # Step 2: Ion current density (requires the particle velocties)
        
        # Compute A^{1}, A^{2}, A^{3} at the particle location
        #
        # Only the last component in retained...
        #A1_p = gather_2D(A[0,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy)
        #A2_p = gather_2D(A[1,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy)
        #A3_p = gather_2D(A[2,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy)
        A1_p = 0.0
        A2_p = 0.0
        A3_p = gather_2D(A3[:,:], x1_ions[i], x2_ions[i], x, y, dx, dy)
        
        # Convert the generalized momentum (denoted here as q) to velocity v
        v1_p = ( p1_ions[i] - q_ions*mu_r*A1_p )/r_ions
        v2_p = ( p2_ions[i] - q_ions*mu_r*A2_p )/r_ions
        v3_p = ( p3_ions[i] - q_ions*mu_r*A3_p )/r_ions
        
        # Calculate the currents on the mesh
        weight1 = weight*v1_p
        weight2 = weight*v2_p
        weight3 = weight*v3_p
        
        # Only the last component in retained...
        #scatter_2D(J_mesh[0,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight1)
        #scatter_2D(J_mesh[1,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight2)
        #scatter_2D(J_mesh[2,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight3)
        scatter_2D(J3_mesh[:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight3)
       
    # End of ion loop
        
    # Scatter electron data to the mesh
    for i in range(N_elec):
        
        # Step 1: Electron charge density
        weight = w_elec*q_elec
        scatter_2D(rho_mesh, x1_elec[i], x2_elec[i], x, y, dx, dy, weight)

        # Step 2: Electron current density (requires the particle velocties)
        
        # Compute A^{1}, A^{2}, A^{3} at the particle location
        #
        # Only the last component in retained...
        #A1_p = gather_2D(A[0,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy)
        #A2_p = gather_2D(A[1,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy)
        #A3_p = gather_2D(A[2,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy)
        A1_p = 0.0
        A2_p = 0.0
        A3_p = gather_2D(A3[:,:], x1_elec[i], x2_elec[i], x, y, dx, dy)
        
        # Convert the generalized momentum (denoted here as q) to velocity v
        v1_p = ( p1_elec[i] - q_elec*mu_r*A1_p )/r_elec
        v2_p = ( p2_elec[i] - q_elec*mu_r*A2_p )/r_elec
        v3_p = ( p3_elec[i] - q_elec*mu_r*A3_p )/r_elec
        
        # Calculate the currents on the mesh
        weight1 = weight*v1_p
        weight2 = weight*v2_p
        weight3 = weight*v3_p
        
        # Only the last component in retained...
        #scatter_2D(J_mesh[0,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight1)
        #scatter_2D(J_mesh[1,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight2)
        #scatter_2D(J_mesh[2,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight3)
        scatter_2D(J3_mesh[:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight3)
        
    # End of electron loop
    
    rho_mesh[:,:] /= cell_volumes[:,:]
    J3_mesh[:,:] /= cell_volumes[:,:]
    
    #J_mesh[0,:,:] /= cell_volumes[:,:]
    #J_mesh[1,:,:] /= cell_volumes[:,:]
    #J_mesh[2,:,:] /= cell_volumes[:,:]
    
    
    # BCs are not periodic
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64, nb.float64[:,:], nb.float64, nb.float64)], 
                  cache = True, boundscheck=False)
def map_rho_to_mesh_2D(rho_mesh, x, y, dx, dy,
                       x1_ions, x2_ions, x1_elec, x2_elec,
                       q_ions, q_elec, cell_volumes, w_ions, w_elec):
    """
    Computes the charge density on the mesh using linear splines.
    """
    
    # Number of simulation particles for each species
    N_ions = x1_ions.size
    N_elec = x1_elec.size
 
    # Clear the contents of the mesh arrays
    rho_mesh[:,:] = 0.0
    
    weight = w_ions*q_ions
    
    # Scatter ion data to the mesh
    for i in range(N_ions):
        
        # Step 1: Ion charge density
        scatter_2D(rho_mesh[:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight)
       
    # End of ion loop
    
    weight = w_elec*q_elec
        
    # Scatter electron data to the mesh
    for i in range(N_elec):
        
        # Step 1: Electron charge density
        scatter_2D(rho_mesh[:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight)
        
    # End of electron loop
    
    rho_mesh[:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None


@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64, nb.float64[:,:], nb.float64, nb.float64)], 
                  cache = True, boundscheck=False)
def map_J_to_mesh_2D3V(J_mesh, x, y, dx, dy,
                       x1_ions, x2_ions, v1_ions, v2_ions, v3_ions,
                       x1_elec, x2_elec, v1_elec, v2_elec, v3_elec,
                       q_ions, q_elec, cell_volumes, w_ions, w_elec):
    """
    Computes the current density for the field solvers using velocity information
    rather than generalized momenta.
    
    This mapping is to be used for the Bennett pinch problem.
    """
    
    # Number of simulation particles for each species
    N_ions = x1_ions.size
    N_elec = x1_elec.size
 
    # Clear the contents of the mesh arrays
    J_mesh[:,:,:] = 0.0
    
    weight = w_ions*q_ions
    
    # Scatter ion data to the mesh
    for i in range(N_ions):
        
        # Step 1: Ion current density (requires the particle velocties)
        
        # Calculate the currents on the mesh
        weight1 = weight*v1_ions[i]
        weight2 = weight*v2_ions[i]
        weight3 = weight*v3_ions[i]
        
        # Only the last component in retained...
        scatter_2D(J_mesh[2,:,:], x1_ions[i], x2_ions[i], x, y, dx, dy, weight3)
       
    # End of ion loop
    
    weight = w_elec*q_elec
        
    # Scatter electron data to the mesh
    for i in range(N_elec):
        
        # Step 1: Electron current density (requires the particle velocties)
        
        # Calculate the currents on the mesh
        weight1 = weight*v1_elec[i]
        weight2 = weight*v2_elec[i]
        weight3 = weight*v3_elec[i]
        
        # Only the last component in retained...
        scatter_2D(J_mesh[2,:,:], x1_elec[i], x2_elec[i], x, y, dx, dy, weight3)
        
    # End of electron loop
    
    J_mesh[0,:,:] /= cell_volumes[:,:]
    J_mesh[1,:,:] /= cell_volumes[:,:]
    J_mesh[2,:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None


# A basic particle push function (2D-3V)
@nb.njit([nb.void(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64)], 
                  cache = True)
def particle_push_2D3P_tao(y1_s, y2_s, p1_s, p2_s, p3_s,
                           x1_s, x2_s, q1_s, q2_s, q3_s,
                           ddx_psi, ddy_psi, A3, ddx_A3, ddy_A3,
                           x, y, dx, dy, q_s, r_s, mu_r, dt):
    """
    Performs half the complete particle integration for a species "s" 
    using field data on the mesh.
    
    It uses all of the particle data for species "s" and updates the positions through
    some step size "dt". The input entries can be interchanged to perform the other half step,
    as well. In this instance of Tao's algorithm, this can be thought of as a particular flow map.
    
    This push is written for a 2D-3P formulation. Note that we have sliced A along time, so that
    it is now a 3D view of the original 4D array.
    """

    # Retrieve the number of particles of species s
    N_p = x1_s.size
    
    for i in range(N_p):
        
        # Step 1: Map the fields at time level "n" to a given particle 
        #
        # These will be needed for the subsequent particle push
        
        # Partial derivatives of the scalar potential
        ddx_psi_p = gather_2D(ddx_psi, x1_s[i], x2_s[i], x, y, dx, dy) 
        ddy_psi_p = gather_2D(ddy_psi, x1_s[i], x2_s[i], x, y, dx, dy)

        # Vector potentials A^{(1)}, A^{(2)}, and A^{(3)}
        #
        # Only the last component in retained...
        #A1_p = gather_2D(A[0,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        #A2_p = gather_2D(A[1,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        #A3_p = gather_2D(A[2,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        A1_p = 0.0
        A2_p = 0.0
        A3_p = gather_2D(A3[:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        
        # Partial derivatives of the vector potential A^{(1)}
        #ddx_A1_p = gather_2D(ddx_A[0,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        #ddy_A1_p = gather_2D(ddy_A[0,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        ddx_A1_p = 0.0
        ddy_A1_p = 0.0

        # Partial derivatives of the vector potential A^{(2)}
        #ddx_A2_p = gather_2D(ddx_A[1,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        #ddy_A2_p = gather_2D(ddy_A[1,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        ddx_A2_p = 0.0
        ddy_A2_p = 0.0
        
        # Partial derivatives of the vector potential A^{(3)}
        #ddx_A3_p = gather_2D(ddx_A[2,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        #ddy_A3_p = gather_2D(ddy_A[2,:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        ddx_A3_p = gather_2D(ddx_A3[:,:], x1_s[i], x2_s[i], x, y, dx, dy)
        ddy_A3_p = gather_2D(ddy_A3[:,:], x1_s[i], x2_s[i], x, y, dx, dy)

        # Step 2: Push the particles using the field data
        #
        # Note: The rhs of (y,p) equations use (x,q)
        # The input entries can be interchanged for the update of (x,q) later, i.e.,
        # the secondary flow map.
        
        # Compute the velocity of the particle
        v1_p = (1/r_s)*( q1_s[i] - q_s*mu_r*A1_p )
        v2_p = (1/r_s)*( q2_s[i] - q_s*mu_r*A2_p )
        v3_p = (1/r_s)*( q3_s[i] - q_s*mu_r*A3_p )
        
        # Update the position
        y1_s[i] += dt*v1_p
        y2_s[i] += dt*v2_p
    
        # Compute the momentum rhs terms
        # This is modified to include contributions from A3 and the corresponding velocity
        F1_s = -q_s*ddx_psi_p + (q_s*mu_r)*(ddx_A1_p*v1_p + ddx_A2_p*v2_p + ddx_A3_p*v3_p)
        F2_s = -q_s*ddy_psi_p + (q_s*mu_r)*(ddy_A1_p*v1_p + ddy_A2_p*v2_p + ddy_A3_p*v3_p)
        # In 2D, no terms show up for F3_s, so we drop this element of the rhs
    
        # Momentum update (contributions for the rhs go away in this problem)
        p1_s[i] += dt*F1_s
        p2_s[i] += dt*F2_s

    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64)], cache = True)
def compute_tao_coupling(x1_s, x2_s, p1_s, p2_s, p3_s,
                         y1_s, y2_s, q1_s, q2_s, q3_s,
                         omega, dt):
    """
    Applies the flow map for omega*H_C for dt time. This is extended to handle 2D-3P
    """
    # Retrieve the number of particles of species s
    N_p = x1_s.size
    
    # Rotation matrix entries for the mixing step
    # This is fixed for second order...
    R_00 =  np.cos(2*omega*dt)
    R_01 =  np.sin(2*omega*dt)
    R_10 = -np.sin(2*omega*dt)
    R_11 =  np.cos(2*omega*dt)
    
    for i in range(N_p):
        
        # Compute the corrections for this particle (accounts for the 2D-3P model)
        x1_bar = 0.5*( (x1_s[i] + y1_s[i]) + R_00*(x1_s[i] - y1_s[i]) + R_01*(p1_s[i] - q1_s[i]) )
        x2_bar = 0.5*( (x2_s[i] + y2_s[i]) + R_00*(x2_s[i] - y2_s[i]) + R_01*(p2_s[i] - q2_s[i]) )

        p1_bar = 0.5*( (p1_s[i] + q1_s[i]) + R_10*(x1_s[i] - y1_s[i]) + R_11*(p1_s[i] - q1_s[i]) )
        p2_bar = 0.5*( (p2_s[i] + q2_s[i]) + R_10*(x2_s[i] - y2_s[i]) + R_11*(p2_s[i] - q2_s[i]) )
        p3_bar = 0.5*( (p3_s[i] + q3_s[i])                            + R_11*(p3_s[i] - q3_s[i]) )
        
        y1_bar = 0.5*( (x1_s[i] + y1_s[i]) - R_00*(x1_s[i] - y1_s[i]) - R_01*(p1_s[i] - q1_s[i]) )
        y2_bar = 0.5*( (x2_s[i] + y2_s[i]) - R_00*(x2_s[i] - y2_s[i]) - R_01*(p2_s[i] - q2_s[i]) )

        q1_bar = 0.5*( (p1_s[i] + q1_s[i]) - R_10*(x1_s[i] - y1_s[i]) - R_11*(p1_s[i] - q1_s[i]) )
        q2_bar = 0.5*( (p2_s[i] + q2_s[i]) - R_10*(x2_s[i] - y2_s[i]) - R_11*(p2_s[i] - q2_s[i]) )
        q3_bar = 0.5*( (p3_s[i] + q3_s[i])                            - R_11*(p3_s[i] - q3_s[i]) )
        
        # Transfer the coupled values back to the particle arrays
        x1_s[i] = x1_bar
        x2_s[i] = x2_bar
        
        p1_s[i] = p1_bar
        p2_s[i] = p2_bar
        p3_s[i] = p3_bar

        y1_s[i] = y1_bar
        y2_s[i] = y2_bar
        
        q1_s[i] = q1_bar
        q2_s[i] = q2_bar
        q3_s[i] = q3_bar

    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64)], 
                  cache=True, boundscheck=False)
def advance_particle_positions_2D(x1_s_new, x2_s_new,
                                  x1_s_old, x2_s_old,
                                  v1_s_mid, v2_s_mid, dt):
    """
    Updates particle positions using the mid-point rule.
    
    This function should be used with the Boris method. 
    """
    
    # Number of particles of a species s
    N_s = x1_s_old.size
    
    for i in range(N_s):
        
        x1_s_new[i] = x1_s_old[i] + dt*v1_s_mid[i] 
        x2_s_new[i] = x2_s_old[i] + dt*v2_s_mid[i]
    
    return None

@nb.njit([nb.float64[:](nb.float64[:], nb.float64[:])], 
          cache=True, boundscheck=False)
def cross_product(a, b):
    """
    Helper function to compute the cross product of two vectors
    in \mathbb{R}^{3}:
    
    a \times b = [a2*b3 - a3*b2]*i + [-a1*b3 + a3*b1]*j + [a1*b2 - a2*b1]*k 
    """
    
    a_cross_b = np.zeros_like(a)
    
    a_cross_b[0] = a[1]*b[2] - a[2]*b[1]
    a_cross_b[1] = a[2]*b[0] - a[0]*b[2]
    a_cross_b[2] = a[0]*b[1] - a[1]*b[0]
    
    return a_cross_b


@nb.njit([nb.void(nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64, nb.float64)], 
                  cache=True, boundscheck=False)
def boris_velocity_push_2D3V(v1_s_new, v2_s_new, v3_s_new,
                             x1_s_old, x2_s_old,
                             v1_s_old, v2_s_old, v3_s_old,
                             E1_mesh, E2_mesh, B1_mesh, B2_mesh,
                             x, y, dx, dy, q_s, r_s, sigma_1, sigma_2, dt):
    """
    Applies a single step of the Boris push to 2D-3V particle data.
    
    Data is passed by reference, so there is no return. Can be
    called for all particles of a certain species. Assumes that the
    position and velocity are offset by dt/2.
    
    Note: This function is specific to this problem.
    """
    
    # Number of particles of a species s
    N_s = x1_s_old.size
    
    # Storage for data at the particles
    E_p = np.empty((3))
    B_p = np.empty((3))
    
    # Other variables for the Boris rotation
    v_minus_vec = np.empty((3))
    v_prime_vec = np.empty((3))
    v_plus_vec  = np.empty((3))
    
    t_vec = np.empty((3)) # t = q*B*dt/(2m)
    s_vec = np.empty((3)) # s = 2t/(1*t^2)
    
    for i in range(N_s):
        
        # First, we need to map the fields from the mesh to the particle
        # using the gather function
        E_p[0] = gather_2D(E1_mesh, x1_s_old[i], x2_s_old[i], x, y, dx, dy)
        E_p[1] = gather_2D(E2_mesh, x1_s_old[i], x2_s_old[i], x, y, dx, dy)
        E_p[2] = 0.0 # E has no z-component for this problem

        B_p[0] = gather_2D(B1_mesh, x1_s_old[i], x2_s_old[i], x, y, dx, dy)
        B_p[1] = gather_2D(B2_mesh, x1_s_old[i], x2_s_old[i], x, y, dx, dy)
        B_p[2] = 0.0 # B has no z-component for this problem
        
        # Compute v_minus = v_old + 0.5*dt*qE/m
        # This problem only uses E^{(1)} and E^{(2)}
        v_minus_vec[0] = v1_s_old[i] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[0])
        v_minus_vec[1] = v2_s_old[i] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[1])
        v_minus_vec[2] = v3_s_old[i] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[2])
        
        # Compute v_prime = v_minus + v_minus \times t 
        t_vec[0] = 0.5*(q_s/r_s)*(sigma_2*B_p[0])*dt
        t_vec[1] = 0.5*(q_s/r_s)*(sigma_2*B_p[1])*dt
        t_vec[2] = 0.5*(q_s/r_s)*(sigma_2*B_p[2])*dt

        # Can use the cross product helper function or inline
        #v_minus_cross_t_vec = cross_product(v_minus_vec, t_vec)
        #v_prime_vec = v_minus_vec + v_minus_cross_t_vec
        
        v_prime_vec[0] = v_minus_vec[0] + (v_minus_vec[1]*t_vec[2] - v_minus_vec[2]*t_vec[1])
        v_prime_vec[1] = v_minus_vec[1] + (v_minus_vec[2]*t_vec[0] - v_minus_vec[0]*t_vec[2])
        v_prime_vec[2] = v_minus_vec[2] + (v_minus_vec[0]*t_vec[1] - v_minus_vec[1]*t_vec[0])
        
        # Compute v_plus = v_minus + v_prime \times s
        t_mag2 = t_vec[0]**2 + t_vec[1]**2 + t_vec[2]**2
        s_vec[0] = 2*t_vec[0]/(1 + t_mag2)
        s_vec[1] = 2*t_vec[1]/(1 + t_mag2)
        s_vec[2] = 2*t_vec[2]/(1 + t_mag2)
        
        # Can use the cross product helper function or inline
        #v_prime_cross_s_vec = cross_product(v_prime_vec, s_vec)
        #v_plus_vec = v_minus_vec + v_prime_cross_s_vec
        
        v_plus_vec[0] = v_minus_vec[0] + (v_prime_vec[1]*s_vec[2] - v_prime_vec[2]*s_vec[1])
        v_plus_vec[1] = v_minus_vec[1] + (v_prime_vec[2]*s_vec[0] - v_prime_vec[0]*s_vec[2])
        v_plus_vec[2] = v_minus_vec[2] + (v_prime_vec[0]*s_vec[1] - v_prime_vec[1]*s_vec[0])
        
        # Compute v_new = v_plus + 0.5*dt*qE/m 
        # This problem only uses E^{(1)} and E^{(2)}
        v1_s_new[i] = v_plus_vec[0] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[0])
        v2_s_new[i] = v_plus_vec[1] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[1])
        v3_s_new[i] = v_plus_vec[2] + 0.5*dt*(q_s/r_s)*(sigma_1*E_p[2])
        
    return None


def apply_particle_BC(x1_s, x2_s, n0, b, alpha, L_x, L_y, L = 1.0):
    """
    "Re-injects" the particles, which leave the domain back into the beam, without
    changing the velocity. The current remains unchanged.
    """
    
    # First find the logical locations of particles
    # that have left the domain
    out_x1_left = np.argwhere(x1_s < -L_x/2)
    out_x1_rite = np.argwhere(x1_s >  L_x/2)
    
    out_x2_left = np.argwhere(x2_s < -L_y/2)
    out_x2_rite = np.argwhere(x2_s >  L_y/2)
    
    # Concatenate all logical locations together
    out_all_idx = np.concatenate((out_x1_left, out_x1_rite, out_x2_left, out_x2_rite))
    
    # Next, find the elements unique to each of these
    # These particle positions will need to be resampled
    out_idx = np.unique(out_all_idx)
    
    N_new = out_idx.size
    
    # Array of random angles for the particles
    # sampled uniformly on [0,2pi) (would linspace be better?)
    theta_array = 2*np.pi*np.random.rand(N_new)
    
    # Array of random fractions
    # sampled uniformly on [0, alpha)
    # with alpha < 1
    frac_array = alpha*np.random.rand(N_new)
    
    # Array for the radius of the particles
    # computed via equation 13.3.14
    r = (1/np.sqrt(n0*b))*np.sqrt(frac_array/(1 - frac_array))
    
    # Now correct the radius for the case that
    # particles live outside of the domain
    # [-L_x/2, L_x/2] x [-L_y/2, L_y/2]
    for i in range(N_new):
        
        while r[i] > L_x/2 and r[i] > L_y/2:
            
            frac = alpha*np.random.rand()
            r[i] = (1/np.sqrt(n0*b))*np.sqrt(frac/(1 - frac))
        
        # End while
        
    # Compute the positions in the non-dimensional domain 
    # using polar coordinates (L is the spatial scaling)
    x1_new = (r/L)*np.cos(theta_array)
    x2_new = (r/L)*np.sin(theta_array)
    
    # Modify the appropriate positions in the original array
    x1_s[out_idx] = x1_new
    x2_s[out_idx] = x2_new
    
    return x1_s, x2_s



@nb.njit([nb.void(nb.float64[:,:], nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], cache = True)
def compute_cell_volume_2D(cell_volume, x, y, dx, dy):
    """
    Computes the volume of 2D Cartesian grid cells.
    
    We know that every cell has volume of dx*dy, but
    modifications are required for boundary cells, so
    we modify this. Boundary cells that are not corners
    have half their original volume while corners are
    a fourth of their original volume.
    """
    N_x = cell_volume.shape[0]
    N_y = cell_volume.shape[1]
    
    # Zero out the cell volumes
    cell_volume[:,:] = 0.0

    for i in range(0, N_x):
        for j in range(0, N_y):
            
            # Interior cells
            if (i > 0 and i < N_x - 1) and (j > 0 and j < N_y - 1):
            
                cell_volume[i,j] = dx*dy
                
            # Boundary cell in y but interior in x 
            elif (i > 0 and i < N_x - 1) and (j == 0 or j == N_y - 1):
            
                cell_volume[i,j] = 0.5*dx*dy
                
            # Boundary cell in x but interior in y 
            elif (i == 0 or i == N_x - 1) and (j > 0 and j < N_y - 1):
            
                cell_volume[i,j] = 0.5*dx*dy
                
            # This is a corner cell
            else:
                
                cell_volume[i,j] = 0.25*dx*dy
    
    return None


@nb.jit(forceobj=True)
def uniform_2D_initialization(a_x, b_x, a_y, b_y, Np_x, Np_y, 
                              endpoint_x = False, endpoint_y = False):
    """
    Initializes a 2-D uniform grid of N_p total particles.
    
    Note that in periodic problems, the endpoint is not included (default).
    
    Returns a pair of 1-D arrays x1_s, x2_s containing the positions 
    of particles of species "s".
    """
    # Compute the total number of particles
    N_p = Np_x * Np_y
    
    # Uniformly spaced particle coordinates in the domain
    x_locs = np.linspace(a_x, b_x, Np_x, endpoint = endpoint_x)
    y_locs = np.linspace(a_y, b_y, Np_y, endpoint = endpoint_y)
    
    # Storage for the particle coordinates
    x1_s = np.zeros([N_p])
    x2_s = np.zeros([N_p])
    
    # Initialize the particles with the convention that
    # x2 changes the fastest
    for i in range(Np_x):
        for j in range(Np_y):
        
            x1_s[ Np_y*i + j ] = x_locs[i]
            x2_s[ Np_y*i + j ] = y_locs[j]
    
    return x1_s, x2_s


#
# Eric Wolf's approach for approximating the Bennett distribution 
#
def set_particle_positions(x1_p, x2_p, N_p, 
                           n0, b, alpha, L_x, L_y):
    """
    Function that computes the (x1,x2) positions of N_p
    particles in the Bennett pinch problem according to
    equation 13.3.14 in Bittencourt.
    
    Note that "b" is computed by (13.3.12, Bittencourt), "alpha" 
    is the fraction of particles contained in the
    plasma column, and "n0" is the number density at the
    axis of the column (13.3.14, Bittencourt).
    
    L_x and L_y are the lengths of the box used to
    run the simulation. No particles live outside of this.
    
    These parameters should be consistently computed
    prior to the call.
    """
    
    # Select a fixed seed for reproducibility
    np.random.seed(0)
    
    # Array of random angles for the particles
    # sampled uniformly on [0,2pi) (would linspace be better?)
    theta_p = 2*np.pi*np.random.rand(N_p)
    
    # Array of random fractions
    # sampled uniformly on [0, alpha)
    # with alpha < 1
    frac_p = alpha*np.random.rand(N_p)
    
    # Array for the radius of the particles
    # computed via equation 13.3.14
    r_p = (1/np.sqrt(n0*b))*np.sqrt(frac_p/(1 - frac_p))
    
    # Now correct the radius for the case that
    # particles live outside of the domain
    # [-L_x/2, L_x/2] x [-L_y/2, L_y/2]
    for i in range(N_p):
        
        while r_p[i] > L_x/2 and r_p[i] > L_y/2:
            
            # Resample
            frac = alpha*np.random.rand()
            r_p[i] = (1/np.sqrt(n0*b))*np.sqrt(frac/(1 - frac))
        
        # End while
        
    # Compute the positions in polar coordinates
    x1_p = r_p*np.cos(theta_p)
    x2_p = r_p*np.sin(theta_p)
    
    return x1_p, x2_p


# #
# # Alternative way to sample the Bennett distribution
# #
# def set_particle_positions(x1_p, x2_p, N_p, 
#                            n0, b, alpha, L_x, L_y):
#     """
#     Function that computes the (x1,x2) positions of N_p
#     particles in the Bennett pinch problem according to
#     equation 13.3.14 in Bittencourt.
    
#     Note that "b" is computed by (13.3.12, Bittencourt), "alpha" 
#     is the fraction of particles contained in the
#     plasma column, and "n0" is the number density at the
#     axis of the column (13.3.14, Bittencourt).
    
#     L_x and L_y are the lengths of the box used to
#     run the simulation. No particles live outside of this.
    
#     These parameters should be consistently computed
#     prior to the call.
    
#     Important: This function works with dimensional variables,
#     which are normalized after the call.
#     """
    
#     global R_beam

#     warnings.filterwarnings('ignore', 'The number of calls to function has reached maxfev')
    
#     # Select a fixed seed for reproducibility
#     np.random.seed(0)

#     # Nonlinear equation to solve
#     G = lambda r, u, n0, b: u - (2/np.pi)*( np.arctan(np.sqrt(n0*b)*r) + (np.sqrt(n0*b)*r)/(1 + n0*b*r**2) )
  
#     # Derivative for the Newton method
#     G_prime = lambda r, u, n0, b: ((4/np.pi)*np.sqrt(b/n0))*( n0/(1 + n0*b*r**2) ) 
    
#     #-------------------------------------------------------------------
#     # Step 1: Sample the distribution over free-space
#     #-------------------------------------------------------------------
    
#     # Let u be an array of samples from a uniform distribution
#     u = np.random.rand(N_p)
#     guess = 0.0
    
#     r_p = np.zeros_like(u) 
    
#     # We solve G(r) = 0 for the sample radius
#     for i in range(N_p):
        
#         r_p[i] = fsolve(G, x0 = guess, args=(u[i], n0, b), fprime=G_prime, xtol= 1.0e-6, maxfev=25)
        
#     #-------------------------------------------------------------------
#     # Step 2: For particles not in the domain, we resample them 
#     # They should also be outside of the beam!
#     #-------------------------------------------------------------------            
    
#     # Make sure all the r_p are contained in the domain
#     finished = np.all(r_p < 0.5*L_x) and np.all(r_p < 0.5*L_y)
    
#     # Identify any particles that need to be resampled
#     # these are particles that have left the domain
#     out_x1 = np.argwhere(r_p >=  L_x/2)
#     out_x2 = np.argwhere(r_p >=  L_y/2)
    
#     # Concatenate all logical locations together
#     out_all_idx = np.concatenate((out_x1, out_x2))
    
#     # Next, find the elements unique to each of these
#     # These particle radii will need to be resampled
#     out_idx = np.unique(out_all_idx)
#     N_new = out_idx.size
    
#     # We solve G(r) = 0 for the sample radius
#     for i in range(N_new):
        
#         finished = False # Ensure we enter the loop
        
#         while not finished:
            
#             # Let u be a sample from a uniform distribution
#             u_new = np.random.rand()
#             guess = 0.0
        
#             # Sample the distribution
#             r_p_new = fsolve(G, x0 = guess, args=(u_new, n0, b), fprime=G_prime, 
#                             xtol= 1.0e-6, maxfev=25)
            
#             #r_p_new += R_beam
    
#             # Particles should be outside the beam but inside the domain
#             outside_beam = r_p_new > R_beam
#             is_contained = (r_p_new < 0.5*L_x) and (r_p_new < 0.5*L_y)
            
#             if outside_beam and is_contained:
                
#                 # Update the sample and move to the next particle
#                 r_p[out_idx[i]] = r_p_new
#                 finished = True
#                 break
                
#             else:
            
#                 # Reject and resample for this particle
#                 finished = False
    
#     # Are all the r_p are contained in the domain?
#     all_contained = np.all(r_p < 0.5*L_x) and np.all(r_p < 0.5*L_y)
#     assert all_contained, "Error: particles not contained. Check initialization.\n"

#     # Array of random angles for the particles
#     # sampled uniformly on [0,2pi) (would linspace be better?)
#     theta_p = 2*np.pi*np.random.rand(N_p)
        
#     # Compute the positions in polar coordinates
#     x1_p = r_p*np.cos(theta_p)
#     x2_p = r_p*np.sin(theta_p)
    
#     return x1_p, x2_p


def set_electron_velocities(N_e, v_thermal):
    """
    Function that computes the (v1,v2) velocities of N_p
    electrons using a Maxwellian distribution.
    """
    
    # Select a fixed seed for reproducibility
    np.random.seed(0)

    # Sample the distribution. Here, we take 
    # the mean velocities to be zero and the variance
    # is the thermal velocity
    v1_e = v_thermal*np.random.randn(N_e)
    v2_e = v_thermal*np.random.randn(N_e)
    
    return v1_e, v2_e




