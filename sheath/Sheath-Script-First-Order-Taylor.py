import numpy as np
import numba as nb
import scipy.sparse as sps
import scipy.sparse.linalg as spslinalg
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq
import sys
import os

# Plotting Utilities
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

# Add the path for the modules to the search path
sys.path.append(r'./Common/')
sys.path.append(r'./Functions/')

# Use LaTeX in the plots
plt.rc('text', usetex=True)

from field_advance import *
from particle_funcs import *
from precompile import *
from plotting_funcs import *
from error import *

from asymmetric_euler_particle_solve import asym_euler_particle_solver

# Profiling 
import time
import cProfile
import pstats
import io

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

##############################
# BEGIN Physical Parameters
##############################

# Speed of light
c = 2.99792458e08  # Units of m/s

# Permittivity and permeability of free space
epsilon_0 = 8.854187817e-12 # Units of L^{-3} M^{-1} T^{4} A^{2}
mu_0 = 1.25663706e-06 # Units of MLT^{-2} A^{-2}

# Boltzmann constant in SI units
k_B = 1.38064852e-23 # Units of L^{2} M T^{-2} K^{-1} (energy units)

# Particle species mass parameters
ion_electron_mass_ratio = 10000.0

electron_charge_mass_ratio = -175882008800.0 # Units of C/kg
ion_charge_mass_ratio = -electron_charge_mass_ratio/ion_electron_mass_ratio # Units of C/kg

M_electron = (-1.602e-19)/electron_charge_mass_ratio
M_ion = ion_electron_mass_ratio*M_electron

Q_electron = electron_charge_mass_ratio*M_electron
Q_ion = ion_charge_mass_ratio*M_ion

##############################
# END Physical Parameters
##############################

grid_refinement = [32,64,128,16]
for g in grid_refinement:
    N = g+1
    # Number of grid points to use
    N_x = N
    N_y = N

    tag = str(g) + "x" + str(g)

    print(tag)

    ##############################
    # BEGIN Domain Parameters
    ##############################

    L_x = 32.0
    L_y = 32.0

    a_x = -L_x/2
    b_x = L_x/2

    a_y = -L_y/2
    b_y =  L_y/2

    dx = (b_x - a_x)/(N_x - 1)
    dy = (b_y - a_y)/(N_y - 1)

    # Generate the grid points with the ends included
    # Grid is non-dimensional but can put units back with L
    x = np.linspace(a_x, b_x, N_x, endpoint=True)
    y = np.linspace(a_y, b_y, N_y, endpoint=True)

    # Scale for mass [kg]
    M = M_electron

    # Scale for (electron) charge [C] (keep this as positive)
    Q = 1.602e-19

    # Compute the average macroscopic number density for the plasma [m^{-3}]
    n_bar = 10**13 # number density in [m^-3]

    # Compute the average macroscopic temperature [K] using lam_D and n_bar
    T_bar = 10000 # temperature in Kelvin [K]

    # Angular oscillation frequency [rad/s]
    w_p = np.sqrt( ( n_bar*(Q**2) )/( M*epsilon_0 ) )

    # Debye length [m]
    lam_D = np.sqrt((epsilon_0 * k_B * T_bar)/(n_bar*Q**2))

    # Define the length and time scales from the plasma parameters
    L = lam_D # L has units of [m]
    T = 1/w_p # T has units of [s/rad]

    # Compute the thermal velocity V = lam_D * w_p in units of [m/s]
    V = L/T

    # Normalized speed of light
    kappa = c/V

    # Derived scales for the scalar potential and vector potential
    # Be careful: If T is not the plasma period, then we shall have constants in
    # front of the wave equations that are not necessarily 1
    psi_0 = (M*V**2)/Q
    A_0 = (M*V)/Q

    # Number density of the electrons (same for ions due to quasi-neutrality)
    n0 = n_bar # n0 has units of [m^{-3}]

    # Scales used in the Lorentz force
    # defined in terms of psi_0 and A_0
    E_0 = psi_0/L
    B_0 = A_0/L

    # These are the coefficients on the sources for the wave equations
    sigma_1 = (M*epsilon_0)/(n_bar*(Q*T)**2)
    sigma_2 = (n_bar*mu_0*(Q*L)**2)/M

    # dt = 5*dx/kappa
    dt = dx/(np.sqrt(2)*kappa)
    T_final = 60
    N_steps = int(T_final/dt)

    v_ave_mag = 1

    # Number of particles for each species
    N_p = int(2.5e5)

    ##############################
    # END Domain Parameters
    ##############################

    ##############################
    # BEGIN Code Parameters
    ##############################
    debug = False
    save_results = True # do we save the figures created?
    results_path = "results/" + tag + "/" # where do we save them?
    write_stride = 100 # save results every n timesteps
    ##############################
    # END Code Parameters
    ##############################

    ##############################
    # BEGIN Derived Parameters
    ##############################

    # More setup
    a_x = x[0]
    b_x = x[-1]
    a_y = y[0]
    b_y = y[-1]
        
    np.random.seed(0) # For reproducibility

    x1_ions = np.zeros([N_p])
    x2_ions = np.zeros([N_p])

    x1_elec = np.zeros([N_p])
    x2_elec = np.zeros([N_p])

    ### Sampling approach for particle intialization
    # # Generate a set of uniform samples in the domain for ions and electrons
    xy_min = [a_x, a_y]
    xy_max = [b_x, b_y]

    # # Create a 2-D array where the columns are x1 and x2 position coordinates
    particle_positions_elec = np.random.uniform(low=xy_min, high=xy_max, size=(N_p,2))
    particle_positions_ions = np.random.uniform(low=xy_min, high=xy_max, size=(N_p,2))


    x1_elec[:] = particle_positions_elec[:,0]
    x2_elec[:] = particle_positions_elec[:,1]

    x1_ions[:] = particle_positions_ions[:,0]
    x2_ions[:] = particle_positions_ions[:,1]

    # Normalized masses
    r_ions = M_ion/M
    r_elec = M_electron/M

    # Normalized mass and charge of the particle species (we suppose there are only 2)
    # Sign of the charge is already included in the charge to mass ratio
    q_ions = Q_ion/Q
    q_elec = Q_electron/Q


    # Ions will be stationary for this experiment
    v1_ions = np.zeros([N_p])
    v2_ions = np.zeros([N_p])

    # Sample the electron velocities from a 2-D Maxwellian
    # Result is stored as a 2-D array
    electron_velocities = np.random.randn(N_p,2)

    # Electrons have drift velocity in addition to a thermal velocity
    v1_elec = v_ave_mag*electron_velocities[:,0]
    v2_elec = v_ave_mag*electron_velocities[:,1]

    # Convert velocity to generalized momentum (A = 0 since the total current is zero)
    # This is equivalent to the classical momentum
    P1_ions = v1_ions*r_ions
    P2_ions = v2_ions*r_ions

    P1_elec = v1_elec*r_elec
    P2_elec = v2_elec*r_elec

    # Compute the normalized particle weights
    # L_x and L_y are the non-dimensional domain lengths
    w_ions = (L_x*L_y)/N_p
    w_elec = (L_x*L_y)/N_p

    ##############################
    # END Derived Parameters
    ##############################

    if debug:
        print(" Numerical reference scalings for this configuration:\n")
        print(" L (Max domain length) [m] =", "{:.6e}".format(L), "\n",
            "T (particle crossing time) [s] =", "{:.6e}".format(T), "\n",
            "V (beam injection velocity) [m/s] =", "{:.6e}".format(V), "\n", 
            "n_bar (average number density) [m^{-3}] =", "{:.6e}".format(n_bar))

        print("----------------------------------------------\n")

        print(" Timestepping information:\n")
        print(" N_steps: " + str(N_steps))
        print(" Field CFL:", "{:0.6e}".format(kappa*dt/min(dx,dy)))
        print(" Particle CFL:", "{:0.6e}".format(v_ave_mag*dt/min(dx,dy)),"\n")
            
        print("----------------------------------------------\n")

        print(" Dimensional quantities:\n")
        print(" Domain length in x [m]:", "{:.6e}".format(L*L_x)) # L_x is non-dimensional
        print(" Domain length in y [m]:", "{:.6e}".format(L*L_y)) # L_y is non-dimensional
        print(" Final time [s]: " + "{:.6e}".format(T_final*T))

        # dt and dx are both non-dimensional
        print(" dx [m] =", "{:.6e}".format(L*dx), "\n",
            "dy [m] =", "{:.6e}".format(L*dy),"\n",
            "dt [s] =", "{:.6e}".format(T*dt), "\n")

        print("----------------------------------------------\n")

        print(" Non-dimensional quantities:\n")
        print(" Domain length in x [non-dimensional]:", "{:.6e}".format(L_x))
        print(" Domain length in y [non-dimensional]:", "{:.6e}".format(L_y))
        print(" v_ave_mag/c:", "{:.6e}".format(V*v_ave_mag/c)) # v_injection is scaled by V
        print(" kappa [non-dimensional] =", "{:.6e}".format(kappa))
        print(" Final time [non-dimensional]: " + "{:.6e}".format(T_final))
        print(" sigma_1 [non-dimensional] =", "{:.6e}".format(sigma_1))
        print(" sigma_2 [non-dimensional] =", "{:.6e}".format(sigma_2))
        print(" dx [non-dimensional] =", 
            "{:.6e}".format(dx), "\n", "dy [non-dimensional] =", 
            "{:.6e}".format(dy), "\n",
            "dt [non-dimensional] =", "{:.6e}".format(dt), "\n")

        # Is the time step small enough?
        assert dt < dx/6, "Make dt smaller. Use more steps or run to a shorter final time.\n"

    total_time, gauge_error, gauss_law_error, sum_gauss_law_residual, N_elec_history, v_elec_var_history = asym_euler_particle_solver(
                                                            x1_ions, x2_ions, 
                                                            P1_ions, P2_ions, 
                                                            v1_ions, v2_ions,
                                                            x1_elec, x2_elec, 
                                                            P1_elec, P2_elec, 
                                                            v1_elec, v2_elec,
                                                            x, y, dx, dy, kappa, T_final, N_steps,
                                                            q_elec, r_elec, w_elec,
                                                            q_ions, r_ions, w_ions,
                                                            sigma_1, sigma_2,
                                                            results_path,
                                                            save_results, write_stride)

# (x1_ions_in, x2_ions_in,
# P1_ions_in, P2_ions_in,
# v1_ions_in, v2_ions_in,
# x1_elec_in, x2_elec_in,
# P1_elec_in, P2_elec_in, 
# v1_elec_in, v2_elec_in,
# x, y, dx, dy, kappa, T_final, N_steps,
# q_elec, r_elec, w_elec,
# q_ions, r_ions, w_ions,
# sigma_1, sigma_2,
# results_path,
# enable_plots = True, plot_at = 500):