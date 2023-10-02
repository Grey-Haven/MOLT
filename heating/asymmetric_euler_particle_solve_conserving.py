import numpy as np
import numba as nb
import scipy.sparse as sps
import scipy.sparse.linalg as spslinalg
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq
import sys
#import pyvie as pv

# Plotting Utilities
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

# Add the path for the modules to the search path
sys.path.append(r'../Common/')
sys.path.append(r'../Functions/')

# Use LaTeX in the plots
# plt.rc('text', usetex=True)

from field_advance import *
from particle_funcs import *
from precompile import *
from plotting_funcs import *
from error import *

# Profiling 
import time
import cProfile
import pstats
import io
import os

print("\n=====================================================================\n")

# How many threads are being used?
print("get_num_threads():", nb.get_num_threads(), "\n")

# Number of CPU cores on the system
print("NUMBA_DEFAULT_NUM_THREADS:", nb.config.NUMBA_DEFAULT_NUM_THREADS, "\n")

# Number of threads n must be less than or equal
# to NUMBA_NUM_THREADS the total number of threads 
# that are launched (defaults to # of physical cores)
print("NUMBA_NUM_THREADS:", nb.config.NUMBA_NUM_THREADS, "\n")

# Use the OpenMP thread backend
# This can be done by setting the
# environment variable NUMBA_THREADING_LAYER="omp"

# Check that the backend is set by the environment variable
print("THREADING_LAYER:", nb.config.THREADING_LAYER, "\n")

@nb.njit([nb.void(nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64, nb.float64, nb.float64)], 
                  cache=True, boundscheck=False)
def improved_asym_euler_momentum_push_2D2P(P1_s_new, P2_s_new,
                                           v1_s_new, v2_s_new,
                                           x1_s_new, x2_s_new,
                                           P1_s_old, P2_s_old,
                                           v1_s_old, v2_s_old,
                                           v1_s_nm1, v2_s_nm1, # Needed for the Taylor approx.
                                           ddx_psi_mesh, ddy_psi_mesh, 
                                           A1_mesh, ddx_A1_mesh, ddy_A1_mesh, 
                                           A2_mesh, ddx_A2_mesh, ddy_A2_mesh, 
                                           x, y, dx, dy, q_s, r_s, dt):
    """
    Applies a single step of the asymmetrical Euler method to 2D-2P particle data.
    
    Data is passed by reference, so there is no return. The new particle position data
    is used to map the fields to the particles. Therefore, the position update needs
    to be done before the momentum push.
    
    Note: This function is specific to the expanding beam problem.
    """
    
    # Number of particles of a species s
    N_s = x1_s_new.size
        
    for i in range(N_s):
        
        # First, we need to map the fields from the mesh to the particle
        # using the gather function based on the new particle coordinates.

        # Scalar potential data
        ddx_psi_p = gather_2D(ddx_psi_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        ddy_psi_p = gather_2D(ddy_psi_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        
        # Vector potential data
        
        # A1
        A1_p = gather_2D(A1_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        ddx_A1_p = gather_2D(ddx_A1_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        ddy_A1_p = gather_2D(ddy_A1_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        
        # A2
        A2_p = gather_2D(A2_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        ddx_A2_p = gather_2D(ddx_A2_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        ddy_A2_p = gather_2D(ddy_A2_mesh, x1_s_new[i], x2_s_new[i], x, y, dx, dy)
        
        # A3 is zero for this problem (so are its derivatives)
        
        # Compute the momentum rhs terms using a Taylor approximation of v^{n+1}
        # that retains the linear terms
        v1_s_star = v1_s_old[i] + ( v1_s_old[i] - v1_s_nm1[i] )
        v2_s_star = v2_s_old[i] + ( v2_s_old[i] - v2_s_nm1[i] )
        rhs1 = -q_s*ddx_psi_p + q_s*( ddx_A1_p*v1_s_star + ddx_A2_p*v2_s_star )
        rhs2 = -q_s*ddy_psi_p + q_s*( ddy_A1_p*v1_s_star + ddy_A2_p*v2_s_star )
        
        # Compute the new momentum
        P1_s_new[i] = P1_s_old[i] + dt*rhs1
        P2_s_new[i] = P2_s_old[i] + dt*rhs2
        
        # Compute the new velocity using the updated momentum
        v1_s_new[i] = (1/r_s)*(P1_s_new[i] - q_s*A1_p)
        v2_s_new[i] = (1/r_s)*(P2_s_new[i] - q_s*A2_p)
        
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64)], 
          cache = True, boundscheck = False)
def compute_ddy_FD(dudy, u, dy):
    """
    Computes an y derivative via finite differences
    """
    
    N_x = u.shape[0]
    N_y = u.shape[1]
    
    # Left boundary (forward diff)
    for i in range(N_x):
        dudy[i,0] = ( -3*u[i,0] + 4*u[i,1] - u[i,2] )/(2*dy)
        
    # Central derivatives
    for i in range(N_x):
        for j in range(1,N_y-1):
            dudy[i,j] = ( u[i,j+1] - u[i,j-1] )/(2*dy)
            
    # Right boundary (backward diff)
    for i in range(N_x):
        dudy[i,-1] = ( 3*u[i,-1] - 4*u[i,-2] + u[i,-3] )/(2*dy)
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64)], 
          cache = True, boundscheck = False)
def compute_ddx_FD(dudx, u, dx):
    """
    Computes an x derivative via finite differences
    """
    
    N_x = u.shape[0]
    N_y = u.shape[1]

    # Left boundary (forward diff)
    for j in range(N_y):
        dudx[0,j] = ( -3*u[0,j] + 4*u[1,j] - u[2,j] )/(2*dx)
        
    # Central derivatives
    for j in range(N_y):
        for i in range(1,N_x-1):
            dudx[i,j] = ( u[i+1,j] - u[i-1,j] )/(2*dx)
            
    # Right boundary (backward diff)
    for j in range(N_y):
        dudx[-1,j] = ( 3*u[-1,j] - 4*u[-2,j] + u[-3,j] )/(2*dx)
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64, nb.float64, nb.float64)], 
                  cache=False, boundscheck=False)
def map_rho_to_mesh_from_J_2D(rho_mesh, J_mesh, dx, dy, dt):
    """
    Computes the charge density on the mesh using 
    the standard single level spline maps.
    
    Assumes a single species is present
    """
    
    sum_J = 0

    # Scatter particle charge data to the mesh
    for i in range(1,rho_mesh.shape[0]-1):
        for j in range(1,rho_mesh.shape[1]-1):
            ip = (i+1)# % rho_mesh.shape[0]
            im = (i-1)# % rho_mesh.shape[0]
            
            jp = (j+1)# % rho_mesh.shape[1]
            jm = (j-1)# % rho_mesh.shape[1]

            # print(jm,j,jp)
            # print(J_mesh[1,i,jm],J_mesh[1,i,j],J_mesh[1,i,jp])
            # print("----------")

            J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
            J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
            J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
            J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

            J1_diff = J_mesh[0,ip,j] - J_mesh[0,im,j]
            J2_diff = J_mesh[1,i,jp] - J_mesh[1,i,jm]

            # print(np.abs(.5*J1_diff - (J1_i_plus - J1_i_minus)))
            # print(np.abs(.5*J2_diff - (J2_j_plus - J2_j_minus)))

            rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))

            sum_J = sum_J + J1_i_plus - J1_i_minus + J2_j_plus - J2_j_minus
    
    # Edge cases

    for i in range(1,rho_mesh.shape[0]-1):
        ip = (i+1)# % rho_mesh.shape[0]
        im = (i-1)# % rho_mesh.shape[0]

        j = 0
        jp = 1
        jm = rho_mesh.shape[1] - 2 # 0,...,N-1, A[0] = A[N-1], so A[N-2] is to the left of A[0] = A[N-1]

        J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
        J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
        J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
        J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

        rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))

        sum_J = sum_J + J1_i_plus - J1_i_minus + J2_j_plus - J2_j_minus
    
    # for i in range(0,rho_mesh.shape[0]):
    #     ip = (i+1)# % rho_mesh.shape[0]
    #     im = (i-1)# % rho_mesh.shape[0]

    #     j = rho_mesh.shape[1] - 1
    #     jp = 1
    #     jm = j - 1 # 0,...,N-1, A[0] = A[N-1], so A[N-2] is to the left of A[0] = A[N-1]

    #     J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
    #     J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
    #     J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
    #     J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

    #     rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))
    
    for j in range(1,rho_mesh.shape[0]-1):
        jp = (j+1)
        jm = (j-1)

        i = 0
        ip = 1
        im = rho_mesh.shape[0] - 2 # 0,...,N-1, A[0] = A[N-1], so A[N-2] is to the left of A[0] = A[N-1]

        J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
        J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
        J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
        J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

        rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))

        sum_J = sum_J + J1_i_plus - J1_i_minus + J2_j_plus - J2_j_minus
    
    # for j in range(0,rho_mesh.shape[0]):
    #     jp = (j+1)
    #     jm = (j-1)

    #     i = rho_mesh.shape[0] - 1
    #     ip = 1
    #     im = i - 1 # 0,...,N-1, A[0] = A[N-1], so A[N-2] is to the left of A[0] = A[N-1]

    #     J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
    #     J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
    #     J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
    #     J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

        # rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))
    
    # Corner case
    i = 0
    j = 0

    ip = 1
    im = rho_mesh.shape[0]-2
    jp = 1
    jm = rho_mesh.shape[1]-2

    J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
    J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
    J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
    J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

    rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))

    sum_J = sum_J + J1_i_plus - J1_i_minus + J2_j_plus - J2_j_minus



    # print(sum_J)

    rho_mesh[:,-1] = rho_mesh[:,0]
    rho_mesh[-1,:] = rho_mesh[0,:]
        

    return None

@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64[:,:], nb.float64)], 
                  cache=False, boundscheck=False)
def map_J_to_mesh_2D2V(J_mesh, x, y, dx, dy,
                       x1, x2, v1, v2,
                       q_s, cell_volumes, w_s):
    """
    Computes the current density for the field solvers using velocity information
    in the 2D-2V setting.
    
    This mapping is to be used for the expanding beam problem.
    """
    
    # Number of simulation particles
    N_part = x1.size
    
    weight = w_s*q_s
    
    # Scatter current to the mesh
    for i in range(N_part):
        
        weight1 = weight*v1[i]
        weight2 = weight*v2[i]
        
        scatter_2D(J_mesh[0,:,:], x1[i], x2[i], x, y, dx, dy, weight1) # J_x
        scatter_2D(J_mesh[1,:,:], x1[i], x2[i], x, y, dx, dy, weight2) # J_y
       
    # End of particle loop
    
    # Divide by the cell volumes to compute the number density
    # Should be careful for the multi-species case. If this function
    # is called for several species, the division occurs multiple times.
    # For this, we can either: Do the division outside of this function or
    # create a rho for each particle species and apply this function (unchanged).
    J_mesh[0,:,:] /= cell_volumes[:,:]
    J_mesh[1,:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64[:,:], nb.float64)], 
                  cache=False, boundscheck=False)
def map_rho_to_mesh_2D(rho_mesh, x, y, dx, dy,
                       x1, x2,
                       q_s, cell_volumes, w_s):
    """
    Computes the charge density on the mesh using 
    the standard single level spline maps.
    
    Assumes a single species is present
    """
    
    # Number of simulation particles
    N_part = x1.size
    
    weight = w_s*q_s
        
    # Scatter particle charge data to the mesh
    for i in range(N_part):
        
        scatter_2D(rho_mesh[:,:], x1[i], x2[i], x, y, dx, dy, weight)
        
    # End of particle loop
    
    # Divide by the cell volumes to compute the number density
    # Should be careful for the multi-species case. If this function
    # is called for several species, the division occurs multiple times.
    # For this, we can either: Do the division outside of this function or
    # create a rho for each particle species and apply this function (unchanged).
    rho_mesh[:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None

def periodic_shift(x, a, L):
    """Performs an element-wise mod by the domain length "L"
    along a coordinate axis whose left-most point is x = a.
    """
    x -= L*np.floor( (x - a)/L)
    
    return x

@nb.njit([nb.void(nb.float64[:,:])], boundscheck=False)
def enforce_periodicity(F_mesh):
    """
    Enforces periodicity in the mesh quantity.
    
    Helper function for the scattering step in the particle to mesh
    mapping for the grid. In the periodic case, the last row/column of the grid
    is redundant. So, any deposit made there will need to be transfered to the 
    corresponding "left side" of the mesh.
    
    For multicomponent fields, this function can be called component-wise.
    
    Note: Be careful to not double count quantities on the mesh!
    """
    
    # Retrieve grid dimensions
    N_x = F_mesh.shape[0]
    N_y = F_mesh.shape[1]
    
    # Transfer the charges to enforce the periodicty
    #
    # Once the transfer is complete, then the edges
    # are copies to create identical, periodice boundaries
    
    # The code below can be verified directly by calculating
    # the charge density, assuming a unform distribution
    for i in range(N_x):
        
        F_mesh[i,0] += F_mesh[i,-1]

    for j in range(N_y):
        
        F_mesh[0,j] += F_mesh[-1,j]
    
    # Copy the first row/column to the final row/column to enforce periodicity
    for j in range(N_y):
        
        F_mesh[-1,j] = F_mesh[0,j]
    
    for i in range(N_x):
        
        F_mesh[i,-1] = F_mesh[i,0]
    
    return None


def asym_euler_particle_heating_solver(x1_ions_in, x2_ions_in, 
                                       P1_ions_in, P2_ions_in, 
                                       v1_ions_in, v2_ions_in,
                                       x1_elec_in, x2_elec_in, 
                                       P1_elec_in, P2_elec_in, 
                                       v1_elec_in, v2_elec_in,
                                       x, y, dx, dy, kappa, T_final, N_steps, 
                                       q_ions, q_elec, 
                                       r_ions, r_elec,
                                       w_ions, w_elec,
                                       sigma_1, sigma_2,
                                       results_path,
                                       enable_plots = True,
                                       plot_at = 500):
    """
    Particle solver for the 2D-2P heating test that uses the asymmetrical Euler method for particles
    and the MOLT field solvers.
    
    Note that this problem starts out as charge neutral and with a net zero current. Therefore, the
    fields are taken to be zero initially.
    """
    # Make a list for tracking the electron velocity history
    # we use this to compute the temperature outside the solver
    # This variance is an average of the variance in each direction
    v_elec_var_history = []
    
    # Useful for plotting later
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Start the timer
    solver_start_time = time.time()
    
    # Grid dimensions
    N_x = x.size
    N_y = y.size
    
    # Domain lengths
    L_x = x[-1] - x[0]
    L_y = y[-1] - y[0]
    
    # Compute the step size
    dt = T_final/N_steps
    
    # MOLT stability parameter
    # Set for the first-order method
    beta_BDF = 1.0
    
    #------------------------------------------------------------------
    # Storage for the integrator
    #------------------------------------------------------------------

    # Initial position, momentum, and velocity of the particles
    # We copy the input data rather than overwrite it
    # and we store two time levels of history
    #
    # We'll assume that the ions remain stationary
    # so that we only need to update electrons.
    
    # Ion positions
    x1_ions = x1_ions_in.copy()
    x2_ions = x2_ions_in.copy() 
    
    # Ion momenta
    P1_ions = P1_ions_in.copy() 
    P2_ions = P2_ions_in.copy()
    
    # Ion velocities
    v1_ions = v1_ions_in.copy() 
    v2_ions = v2_ions_in.copy() 
    
    # Electron positions
    x1_elec_old = x1_elec_in.copy()
    x2_elec_old = x2_elec_in.copy() 
    
    x1_elec_new = x1_elec_in.copy() 
    x2_elec_new = x2_elec_in.copy()
    
    # Electron momenta
    P1_elec_old = P1_elec_in.copy() 
    P2_elec_old = P2_elec_in.copy()
    
    P1_elec_new = P1_elec_in.copy() 
    P2_elec_new = P2_elec_in.copy()
    
    # Electron velocities
    v1_elec_old = v1_elec_in.copy() 
    v2_elec_old = v2_elec_in.copy()
    
    v1_elec_new = v1_elec_in.copy() 
    v2_elec_new = v2_elec_in.copy()
    
    # Velocity at time t^{n-1} used for the Taylor approx. 
    v1_elec_nm1 = v1_elec_in.copy() 
    v2_elec_nm1 = v2_elec_in.copy()
    
    # Taylor approximated velocity
    # v_star = v^{n} + ddt(v^{n})*dt
    # which is approximated by
    # v^{n} + (v^{n} - v^{n-1})
    v1_elec_star = v1_elec_in.copy() 
    v2_elec_star = v2_elec_in.copy()
    
    # Store the total number of particles for each species
    N_ions = x1_ions_in.size
    N_elec = x1_elec_in.size
    
    # Mesh/field data
    # Need psi, A1, and A2
    # as well as their derivatives
    #
    # We compute ddt_psi with backwards differences
    psi = np.zeros([3,N_x,N_y])
    ddx_psi = np.zeros([N_x,N_y])
    ddy_psi = np.zeros([N_x,N_y])
    psi_src = np.zeros([N_x,N_y])
    
    A1 = np.zeros([3,N_x, N_y])
    ddx_A1 = np.zeros([N_x,N_y])
    ddy_A1 = np.zeros([N_x,N_y])
    A1_src = np.zeros([N_x,N_y])
    
    A2 = np.zeros([3,N_x, N_y])
    ddx_A2 = np.zeros([N_x,N_y])
    ddy_A2 = np.zeros([N_x,N_y])
    A2_src = np.zeros([N_x,N_y])
    
    # Other data needed for the evaluation of 
    # the gauge and Gauss' law
    ddt_psi = np.zeros([N_x,N_y])
    ddt_A1 = np.zeros([N_x,N_y])
    ddt_A2 = np.zeros([N_x,N_y])
    
    E1 = np.zeros([N_x,N_y])
    E2 = np.zeros([N_x,N_y])
    
    # Note that from the relation B = curl(A), we identify
    # B3 = ddx(A2) - ddy(A1)
    B3 = np.zeros([N_x,N_y])
    
    ddx_E1 = np.zeros([N_x,N_y])
    ddy_E2 = np.zeros([N_x,N_y])
    
    gauge_residual = np.zeros([N_x,N_y])
    gauss_law_residual = np.zeros([N_x,N_y])
    
    gauge_error = np.zeros([N_steps])
    gauss_law_error = np.zeros([N_steps])
    sum_gauss_law_residual = np.zeros([N_steps])
    rho_history = np.zeros([N_steps])

    # Storage for the particle data on the mesh
    rho_ions = np.zeros([N_x,N_y])
    rho_elec = np.zeros([N_x,N_y])
    rho_mesh = np.zeros([N_x,N_y])
    
    # We track three time levels of J (n, n+1)
    # Note, we don't need J3 for this model 
    # Since ions are stationary J_mesh := J_elec
    J_mesh = np.zeros([2,N_x,N_y]) # Idx order: comp., grid indices
    
    ddx_J1 = np.zeros([N_x,N_y])
    ddy_J2 = np.zeros([N_x,N_y])
    
    # Compute the cell volumes required in the particle to mesh mapping
    # The domain is periodic here, so the first and last cells here are
    # identical.
    cell_volumes = dx*dy*np.ones([N_x,N_y])
        
    # Current time of the simulation and step counter
    t_n = 0.0
    steps = 0

    csv_path = os.path.join(results_path, "csv_files")
    figures_path = os.path.join(results_path, "figures")

    rho_plot_path = os.path.join(figures_path,"rho-plot")
    J_plot_path = os.path.join(figures_path,"J-plot")
    A1_plot_path = os.path.join(figures_path,"A1-plot")
    A2_plot_path = os.path.join(figures_path,"A2-plot")
    psi_plot_path = os.path.join(figures_path,"phi-plot")
    gauge_slice_plot_path = os.path.join(figures_path,"gauge-plot","slice")
    gauge_surface_plot_path = os.path.join(figures_path,"gauge-plot","surface")
    gauss_slice_plot_path = os.path.join(figures_path,"gauss-plot","slice")
    gauss_surface_plot_path = os.path.join(figures_path,"gauss-plot","surface")
    E_plot_path = os.path.join(figures_path,"E-plot")
    B_plot_path = os.path.join(figures_path,"B-plot")

    if enable_plots:

        results_paths = [rho_plot_path, J_plot_path, A1_plot_path, A2_plot_path,
                        psi_plot_path, gauge_slice_plot_path, gauge_surface_plot_path,
                        gauss_slice_plot_path, gauss_surface_plot_path, E_plot_path, B_plot_path]
        for path in results_paths:
            if ~os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    electron_csv_path = csv_path+"electron-csv/"
    rho_csv_path = csv_path+"rho-csv/"
    J1_csv_path = csv_path+"J1-csv/"
    J2_csv_path = csv_path+"J2-csv/"
    A1_csv_path = csv_path+"A1-csv/"
    A2_csv_path = csv_path+"A2-csv/"
    psi_csv_path = csv_path+"phi-csv/"
    gauge_csv_path = csv_path+"gauge-csv/"
    gauss_csv_path = csv_path+"gauss-csv/"
    E1_csv_path = csv_path+"E1-csv/"
    E2_csv_path = csv_path+"E2-csv/"
    B3_csv_path = csv_path+"B3-csv/"

    # Ions
    map_rho_to_mesh_2D(rho_ions[:,:], x, y, dx, dy,
                    x1_ions, x2_ions,
                    q_ions, cell_volumes, w_ions)

    # Electrons
    map_rho_to_mesh_2D(rho_elec[:,:], x, y, dx, dy,
                    x1_elec_new, x2_elec_new,
                    q_elec, cell_volumes, w_elec)
    # Need to enforce periodicity for the charge on the mesh
    enforce_periodicity(rho_ions[:,:])
    enforce_periodicity(rho_elec[:,:])

    rho_mesh = rho_ions + rho_elec
    
    while(steps < N_steps):

        #---------------------------------------------------------------------
        # 1. Advance electron positions by dt using v^{n}
        #---------------------------------------------------------------------
         
        advance_particle_positions_2D(x1_elec_new, x2_elec_new,
                                      x1_elec_old, x2_elec_old,
                                      v1_elec_old, v2_elec_old, dt)
        
        # Apply the particle boundary conditions
        # Need to include the shift function here
        periodic_shift(x1_elec_new, x[0], L_x)
        periodic_shift(x2_elec_new, y[0], L_y)

        #---------------------------------------------------------------------
        # 2. Compute the electron current density used for updating A
        #---------------------------------------------------------------------

        # Clear the contents of J prior to the mapping
        # This is done here b/c the J function does not reset the current
        # We do this so that it can be applied to any number of species
        
        J_mesh[:,:,:] = 0.0
        
        # Map for electrons (ions are stationary)
        # Can try using the starred velocities here if we want
        map_J_to_mesh_2D2V(J_mesh[:,:,:], x, y, dx, dy,
                           x1_elec_new, x2_elec_new, 
                           v1_elec_old, v2_elec_old,
                           q_elec, cell_volumes, w_elec)
        

        # Need to enforce periodicity for the current on the mesh
        enforce_periodicity(J_mesh[0,:,:])
        enforce_periodicity(J_mesh[1,:,:])

        assert all(J_mesh[0,0,:] == J_mesh[0,-1,:])
        assert all(J_mesh[0,:,0] == J_mesh[0,:,-1])

        assert all(J_mesh[1,0,:] == J_mesh[1,-1,:])
        assert all(J_mesh[1,:,0] == J_mesh[1,:,-1])
        
        
        # Compute components of div(J) using finite-differences
        compute_ddx_FD(ddx_J1, J_mesh[0,:,:], dx)
        compute_ddy_FD(ddy_J2, J_mesh[1,:,:], dy)
        
        #---------------------------------------------------------------------
        # 4. Using the new positions, map charge to the mesh to get rho^{n+1}
        #---------------------------------------------------------------------
        
        # Clear the contents of rho at time level n+1
        # prior to the mapping
        # This is done here b/c the function does not reset the current
        # We do this so that it can be applied to any number of species
        # rho_ions[:,:] = 0.0
        # rho_elec[:,:] = 0.0

        # Ions
        # map_rho_to_mesh_2D(rho_ions[:,:], x, y, dx, dy,
        #                    x1_ions, x2_ions,
        #                    q_ions, cell_volumes, w_ions)
        
        # Electrons
        # map_rho_to_mesh_2D(rho_elec[:,:], x, y, dx, dy,
        #                    x1_elec_new, x2_elec_new,
        #                    q_elec, cell_volumes, w_elec)

        map_rho_to_mesh_from_J_2D(rho_elec, J_mesh, dx, dy, dt)

        # Need to enforce periodicity for the charge on the mesh
        # enforce_periodicity(rho_ions[:,:])
        # enforce_periodicity(rho_elec[:,:])

        # Compute the total charge density
        rho_mesh[:,:] = rho_ions[:,:] + rho_elec[:,:]
    
        assert all(rho_mesh[0,:] == rho_mesh[-1,:])
        assert all(rho_mesh[:,0] == rho_mesh[:,-1])
        
        #---------------------------------------------------------------------
        # 5. Advance the psi and its derivatives by dt using BDF-1 
        #---------------------------------------------------------------------
        
        psi_src[:,:] = (1/sigma_1)*rho_mesh[:,:]
        
        # Charge density is at the new time level from step (3)
        # which is consistent with the BDF scheme
        BDF1_combined_per_advance(psi, ddx_psi, ddy_psi, psi_src,
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # Wait to shuffle until the end, but we could do that here
        
        #---------------------------------------------------------------------
        # 5. Advance the A1 and A2 and their derivatives by dt using BDF-1
        #---------------------------------------------------------------------
        
        A1_src[:,:] = sigma_2*J_mesh[0,:,:]
        A2_src[:,:] = sigma_2*J_mesh[1,:,:]
        
        # A1 uses J1
        BDF1_combined_per_advance(A1, ddx_A1, ddy_A1, A1_src[:,:],
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # A2 uses J2
        BDF1_combined_per_advance(A2, ddx_A2, ddy_A2, A2_src[:,:],
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # Wait to shuffle until the end, but we could do that here
        
        #---------------------------------------------------------------------
        # 6. Momentum advance by dt
        #---------------------------------------------------------------------
        
        # Fields are taken implicitly and we use the "lagged" velocity
        #
        # This will give us new momenta and velocities for the next step
        improved_asym_euler_momentum_push_2D2P(P1_elec_new, P2_elec_new,
                                               v1_elec_new, v2_elec_new,
                                               x1_elec_new, x2_elec_new,
                                               P1_elec_old, P2_elec_old,
                                               v1_elec_old, v2_elec_old,
                                               v1_elec_nm1, v2_elec_nm1,
                                               ddx_psi, ddy_psi, 
                                               A1[-1], ddx_A1, ddy_A1, 
                                               A2[-1], ddx_A2, ddy_A2, 
                                               x, y, dx, dy, q_elec, r_elec, dt)
        
        #---------------------------------------------------------------------
        # 7. Compute the errors in the Lorenz gauge and Gauss' law
        #---------------------------------------------------------------------
        
        # Compute the time derivative of psi using finite differences
        ddt_psi[:,:] = ( psi[-1,:,:] - psi[-2,:,:] )/dt
        
        # Compute the residual in the Lorenz gauge 
        gauge_residual[:,:] = (1/kappa**2)*ddt_psi[:,:] + ddx_A1[:,:] + ddy_A2[:,:]
        
        gauge_error[steps] = get_L_2_error(gauge_residual[:,:], 
                                           np.zeros_like(gauge_residual[:,:]), 
                                           dx*dy)
        
        rho_history[steps] = np.sum(np.sum(rho_mesh))

        # Compute the ddt_A with backwards finite-differences
        ddt_A1[:,:] = ( A1[-1,:,:] - A1[-2,:,:] )/dt
        ddt_A2[:,:] = ( A2[-1,:,:] - A2[-2,:,:] )/dt
        
        # Compute E = -grad(psi) - ddt_A
        # For ddt A, we use backward finite-differences
        # Note, E3 is not used in the particle update so we don't need ddt_A3
        E1[:,:] = -ddx_psi[:,:] - ddt_A1[:,:]
        E2[:,:] = -ddy_psi[:,:] - ddt_A2[:,:]
        
        # Compute Gauss' law div(E) - rho to check the involution
        # We'll just use finite-differences here
        compute_ddx_FD(ddx_E1, E1, dx)
        compute_ddy_FD(ddy_E2, E2, dy)
        
        gauss_law_residual[:,:] = ddx_E1[:,:] + ddy_E2[:,:] - psi_src[:,:]
        
        gauss_law_error[steps] = get_L_2_error(gauss_law_residual[:,:], 
                                               np.zeros_like(gauss_law_residual[:,:]), 
                                               dx*dy)
        
        # Now we measure the sum of the residual in Gauss' law (avoiding the boundary)
        sum_gauss_law_residual[steps] = np.sum(gauss_law_residual[:,:])
        
        #---------------------------------------------------------------------
        # 8. Prepare for the next time step by shuffling the time history data
        #---------------------------------------------------------------------
        
        # Shuffle the time history of the fields
        shuffle_steps(psi)
        shuffle_steps(A1)
        shuffle_steps(A2)
        
        # Shuffle the time history of the particle data
        v1_elec_nm1[:] = v1_elec_old[:]
        v2_elec_nm1[:] = v2_elec_old[:]
        
        x1_elec_old[:] = x1_elec_new[:] 
        x2_elec_old[:] = x2_elec_new[:]
        
        v1_elec_old[:] = v1_elec_new[:] 
        v2_elec_old[:] = v2_elec_new[:]
        
        P1_elec_old[:] = P1_elec_new[:] 
        P2_elec_old[:] = P2_elec_new[:]
        
        # Measure the variance of the electron velocity distribution
        # and store for later use
        #
        # Note that we average the variance here so we don't need an
        # extra factor of two outside of this function
        var_v1 = np.var(v1_elec_new)
        var_v2 = np.var(v2_elec_new)
        v_elec_var_history.append( 0.5*(var_v1 + var_v2) )
        
        if enable_plots:
            
            # Should also plot things at the final step as well
            if steps % plot_at == 0 or steps + 1 == N_steps:
                
                print("Finished with step:", steps,"\n")
                
                # Don't measure the charge at the redundant boundary points
                print("Total charge:","{:.6e}".format(np.sum(cell_volumes[:-1,:-1]*rho_mesh[:-1,:-1])),"\n")
                print("L2 error for the Gauge:","{:.6e}".format(gauge_error[steps]),"\n")
                print("L2 error for Gauss' law:","{:.6e}".format(gauss_law_error[steps]),"\n")
                print("Sum of the residual for Gauss' law:","{:.6e}".format(sum_gauss_law_residual[steps]),"\n")
                
                # Plot of the particles and charge density
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8), sharex=False, sharey=True)
                
                part_skip = 10 # May not want to plot EVERY particle
                
                axes[0].scatter(x1_elec_new[::part_skip], x2_elec_new[::part_skip], 
                                s = 10.0, c = "b", marker = "o")
                
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"Electron Positions at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                axes[0].grid(linestyle='--')
            
                im = axes[1].pcolormesh(X, Y, rho_mesh[:,:], cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_title( r"$\rho$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout(w_pad=4)
                plt.savefig(os.path.join(rho_plot_path, "Heating-aem1-taylor-electron-rho-plot-step"+str(steps)+".png"), bbox_inches="tight")                
                
                # Plot of the current densities: J1 and J2
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8), sharex=False, sharey=True)
                
                im = axes[0].pcolormesh(X, Y, J_mesh[0,:,:], cmap = 'viridis', shading='auto')
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].xaxis.offsetText.set_fontsize(32)
                axes[0].yaxis.offsetText.set_fontsize(32)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"$J^{(1)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[0])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[1].pcolormesh(X, Y, J_mesh[1,:,:], cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].tick_params(axis='y', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].yaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_ylim((y[0],y[-1]))
                axes[1].set_title( r"$J^{(2)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout()
                plt.savefig(os.path.join(J_plot_path, "Heating-aem1-taylor-J-step"+str(steps)+".png"), bbox_inches="tight")

                
                # Plot of A1, ddx_A1, ddy_A1
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28,8), sharex=False, sharey=True)
                
                im = axes[0].pcolormesh(X, Y, A1[-1,:,:], cmap = 'viridis', shading='auto')
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].xaxis.offsetText.set_fontsize(32)
                axes[0].yaxis.offsetText.set_fontsize(32)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"$A^{(1)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[0])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)

                im = axes[1].pcolormesh(X, Y, ddx_A1, cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].tick_params(axis='y', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].yaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_ylim((y[0],y[-1]))
                axes[1].set_title( r"$\partial_{x} A^{(1)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[2].pcolormesh(X, Y, ddy_A1, cmap = 'viridis', shading='auto')
                axes[2].set_xlabel(r"$x$", fontsize=32)
                axes[2].tick_params(axis='x', labelsize=32, pad=10)
                axes[2].tick_params(axis='y', labelsize=32, pad=10)
                axes[2].xaxis.offsetText.set_fontsize(32)
                axes[2].yaxis.offsetText.set_fontsize(32)
                axes[2].set_xlim((x[0],x[-1]))
                axes[2].set_ylim((y[0],y[-1]))
                axes[2].set_title( r"$\partial_{y} A^{(1)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[2])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout()
                plt.savefig(os.path.join(A1_plot_path, "Heating-aem1-taylor-A1-step"+str(steps)+".png"), bbox_inches="tight")
                
                
                # Plot of A2, ddx_A2, ddy_A2
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28,8), sharex=False, sharey=True)
                
                im = axes[0].pcolormesh(X, Y, A2[-1,:,:], cmap = 'viridis', shading='auto')
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].xaxis.offsetText.set_fontsize(32)
                axes[0].yaxis.offsetText.set_fontsize(32)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"$A^{(2)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[0])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)

                im = axes[1].pcolormesh(X, Y, ddx_A2, cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].tick_params(axis='y', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].yaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_ylim((y[0],y[-1]))
                axes[1].set_title( r"$\partial_{x} A^{(2)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[2].pcolormesh(X, Y, ddy_A2, cmap = 'viridis', shading='auto')
                axes[2].set_xlabel(r"$x$", fontsize=32)
                axes[2].tick_params(axis='x', labelsize=32, pad=10)
                axes[2].tick_params(axis='y', labelsize=32, pad=10)
                axes[2].xaxis.offsetText.set_fontsize(32)
                axes[2].yaxis.offsetText.set_fontsize(32)
                axes[2].set_xlim((x[0],x[-1]))
                axes[2].set_ylim((y[0],y[-1]))
                axes[2].set_title( r"$\partial_{y} A^{(2)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[2])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout()
                plt.savefig(os.path.join(A2_plot_path, "Heating-aem1-taylor-A2-step"+str(steps)+".png"), bbox_inches="tight")

                
                # Plot of psi, ddx_psi, and ddy_psi
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28,8), sharex=False, sharey=True)
                
                im = axes[0].pcolormesh(X, Y, psi[-1,:,:], cmap = 'viridis', shading='auto')
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].xaxis.offsetText.set_fontsize(32)
                axes[0].yaxis.offsetText.set_fontsize(32)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"$\psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[0])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[1].pcolormesh(X, Y, ddx_psi, cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                #axes[1].set_ylabel(r"$y$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].tick_params(axis='y', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].yaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_ylim((y[0],y[-1]))
                axes[1].set_title( r"$\partial_{x} \psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[2].pcolormesh(X, Y, ddy_psi, cmap = 'viridis', shading='auto')
                axes[2].set_xlabel(r"$x$", fontsize=32)
                #axes[2].set_ylabel(r"$y$", fontsize=32)
                axes[2].tick_params(axis='x', labelsize=32, pad=10)
                axes[2].tick_params(axis='y', labelsize=32, pad=10)
                axes[2].xaxis.offsetText.set_fontsize(32)
                axes[2].yaxis.offsetText.set_fontsize(32)
                axes[2].set_xlim((x[0],x[-1]))
                axes[2].set_ylim((y[0],y[-1]))
                axes[2].set_title( r"$\partial_{y} \psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[2])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout()
                plt.savefig(os.path.join(psi_plot_path, "Heating-aem1-taylor-psi-step"+str(steps)+".png"), bbox_inches="tight")
                

                plt.figure(figsize=(10,8))
                
                plt.plot(x, gauge_residual[:,int(N_y/2)], lw=2.0, ls="-", color="b")
                plt.xlabel(r"$x$", fontsize=32)
                ax = plt.gca()
                ax.set_xlabel(r"$x$", fontsize=32)
                ax.tick_params(axis='x', labelsize=32)
                ax.tick_params(axis='y', labelsize=32)
                ax.xaxis.offsetText.set_fontsize(32)
                ax.yaxis.offsetText.set_fontsize(32)
                ax.grid(ls="--")
                ax.set_xlim((x[0],x[-1])) 
                               
                plt.savefig(gauge_slice_plot_path + "Heating-aem1-taylor-gauge-slice-step"+str(steps)+".png", bbox_inches="tight")
                               
                    
                fig = plt.figure(figsize=(12,10))
                ax = fig.gca(projection='3d')

                # Plot the surface.
                surf = ax.plot_surface(X, Y, gauge_residual, rstride=1, cstride=1, cmap="viridis", edgecolor="none")

                # Axes properties
                ax.set_xlabel(r'$x$', fontsize=40, labelpad=45.0)
                ax.set_ylabel(r'$y$', fontsize=40, labelpad=45.0)

                ax.tick_params(axis='x', labelsize=32, pad=15)
                ax.tick_params(axis='y', labelsize=32, pad=15)
                ax.tick_params(axis='z', labelsize=32, pad=15)

                ax.xaxis.offsetText.set_fontsize(32)
                ax.yaxis.offsetText.set_fontsize(32)
                ax.zaxis.offsetText.set_fontsize(32)

                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.zaxis.set_major_locator(plt.MaxNLocator(5))
                
                ax.set_zticks([])
                
                ax.set_title(r"$\frac{1}{\kappa^2}\partial_{t} \psi + \nabla \cdot \mathbf{A}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                
                cbar = plt.colorbar(surf, shrink=0.8, ax=ax, pad=0.1)
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.savefig(os.path.join(gauge_surface_plot_path, "Heating-aem1-taylor-lorenz-gauge-surface-step"+str(steps)+".png"), bbox_inches="tight")
 
        
                plt.figure(figsize=(10,8))
                
                plt.plot(x, gauss_law_residual[:,int(N_y/2)], lw=2.0, ls="-", color="b")
                plt.xlabel(r"$x$", fontsize=32)
                ax = plt.gca()
                ax.set_xlabel(r"$x$", fontsize=32)
                ax.tick_params(axis='x', labelsize=32)
                ax.tick_params(axis='y', labelsize=32)
                ax.xaxis.offsetText.set_fontsize(32)
                ax.yaxis.offsetText.set_fontsize(32)
                ax.grid(ls="--")
                ax.set_xlim((x[0],x[-1])) 
                               
                plt.savefig(os.path.join(gauss_slice_plot_path, "Heating-aem1-taylor-gauss-law-slice-step"+str(steps)+".png"), bbox_inches="tight")
                    
                fig = plt.figure(figsize=(12,10))
                ax = fig.gca(projection='3d')

                # Plot the surface.
                surf = ax.plot_surface(X, Y, gauss_law_residual, rstride=1, cstride=1, cmap="viridis", edgecolor="none")

                # Axes properties
                ax.set_xlabel(r'$x$', fontsize=40, labelpad=45.0)
                ax.set_ylabel(r'$y$', fontsize=40, labelpad=45.0)
                #ax.set_zlabel(r"Residual", fontsize=40, labelpad=45.0)

                ax.tick_params(axis='x', labelsize=32, pad=15)
                ax.tick_params(axis='y', labelsize=32, pad=15)
                ax.tick_params(axis='z', labelsize=32, pad=15)

                ax.xaxis.offsetText.set_fontsize(32)
                ax.yaxis.offsetText.set_fontsize(32)
                ax.zaxis.offsetText.set_fontsize(32)

                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.zaxis.set_major_locator(plt.MaxNLocator(5))
                ax.set_title(r"$\nabla \cdot \mathbf{E} - \frac{\rho}{\sigma_1}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                
                ax.set_zticks([])
                
                cbar = plt.colorbar(surf, shrink=0.8, ax=ax, pad=0.1)
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.savefig(os.path.join(gauss_surface_plot_path, "Heating-aem1-taylor-gauss-law-surface-step"+str(steps)+".png"), bbox_inches="tight")
                

                
                # Plot of the electric field components E1 and E2
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8), sharex=False, sharey=True)
                
                im = axes[0].pcolormesh(X, Y, E1, cmap = 'viridis', shading='auto')
                axes[0].set_xlabel(r"$x$", fontsize=32)
                axes[0].set_ylabel(r"$y$", fontsize=32)
                axes[0].tick_params(axis='x', labelsize=32, pad=10)
                axes[0].tick_params(axis='y', labelsize=32, pad=10)
                axes[0].xaxis.offsetText.set_fontsize(32)
                axes[0].yaxis.offsetText.set_fontsize(32)
                axes[0].set_xlim((x[0],x[-1]))
                axes[0].set_ylim((y[0],y[-1]))
                axes[0].set_title( r"$E^{(1)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[0])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                im = axes[1].pcolormesh(X, Y, E2, cmap = 'viridis', shading='auto')
                axes[1].set_xlabel(r"$x$", fontsize=32)
                axes[1].tick_params(axis='x', labelsize=32, pad=10)
                axes[1].tick_params(axis='y', labelsize=32, pad=10)
                axes[1].xaxis.offsetText.set_fontsize(32)
                axes[1].yaxis.offsetText.set_fontsize(32)
                axes[1].set_xlim((x[0],x[-1]))
                axes[1].set_ylim((y[0],y[-1]))
                axes[1].set_title( r"$E^{(2)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes[1])
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.tight_layout()
                plt.savefig(os.path.join(E_plot_path, "Heating-aem1-taylor-E-field-step"+str(steps)+".png"), bbox_inches="tight")
                
                # Plot of B3 for comparison
                # Compute the magnetic field entry B3 using the derivatives of A1 and A2
                B3[:,:] =  ddx_A2[:,:] - ddy_A1[:,:]

                # Plot of the electric field components E1 and E2
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8), sharex=False, sharey=False)
                
                im = axes.pcolormesh(X, Y, B3, cmap = 'viridis', shading='auto')
                axes.set_xlabel(r"$x$", fontsize=32)
                axes.set_ylabel(r"$y$", fontsize=32)
                axes.tick_params(axis='x', labelsize=32, pad=10)
                axes.tick_params(axis='y', labelsize=32, pad=10)
                axes.xaxis.offsetText.set_fontsize(32)
                axes.yaxis.offsetText.set_fontsize(32)
                axes.set_xlim((x[0],x[-1]))
                axes.set_ylim((y[0],y[-1]))
                axes.set_title( r"$B^{(3)}$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                cbar = plt.colorbar(im, ax=axes)
                cbar.ax.tick_params(labelsize=32)
                cbar.ax.yaxis.offsetText.set(size=32)
                
                plt.savefig(os.path.join(B_plot_path, "Heating-aem1-taylor-B3-step"+str(steps)+".png"), bbox_inches="tight")
                
                # Close the figures
                plt.close(fig="all")

        # Step is now complete
        steps += 1
        t_n += dt
        
    # Stop the timer
    solver_end_time = time.time()

    total_time = solver_end_time - solver_start_time
    
    return total_time, gauge_error, gauss_law_error, sum_gauss_law_residual, v_elec_var_history, rho_history
