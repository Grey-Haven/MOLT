import numpy as np
import matplotlib.pyplot as plt

import time
import os

from particle_funcs import advance_particle_positions_2D
from remove_particles import remove_particles
from map_fields import map_J_to_mesh_2D2V, map_rho_to_mesh_2D
from derivatives import compute_ddx_FD, compute_ddy_FD
from error import get_L_2_error
from field_advance import BDF1_combined_dir_advance, shuffle_steps
from asymmetric_euler_momentum_push import improved_asym_euler_momentum_push_2D2P

def asym_euler_particle_solver(x1_ions_in, x2_ions_in,
                               P1_ions_in, P2_ions_in,
                               v1_ions_in, v2_ions_in,
                               x1_elec_in, x2_elec_in,
                               P1_elec_in, P2_elec_in, 
                               v1_elec_in, v2_elec_in,
                               x, y, dx, dy, kappa, T_final, N_steps,
                               q_elec, r_elec, w_elec,
                               q_ions, r_ions, w_ions,
                               sigma_1, sigma_2,
                               results_path,
                               enable_plots = True, enable_csvs = True,
                               plot_at = 500):
    """
    Particle solver for the expanding beam that uses the improved asymmetrical Euler method for particles
    and the MOLT field solvers.
    """
    # Make a list for tracking the electron velocity history
    # we use this to compute the temperature outside the solver
    # This variance is an average of the variance in each direction
    v_elec_var_history = []
    
    a_x = x[0]
    b_x = x[-1]
    a_y = y[0]
    b_y = y[-1]
    
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
    
    # Counter for tracking the number of particles
    # in the domain at any given time
    N_elec_now = N_elec
    
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

    # Storage for the particle data on the mesh
    rho_ions = np.zeros([N_x,N_y])
    rho_elec = np.zeros([N_x,N_y])
    rho_mesh = np.zeros([N_x,N_y])
    
    # We track three time levels of J (n, n+1)
    # Note, we don't need J3 for this model    
    J_mesh = np.zeros([2,N_x,N_y]) # Idx order: comp., grid indices
    
    ddx_J1 = np.zeros([N_x,N_y])
    ddy_J2 = np.zeros([N_x,N_y])
    
    cell_volumes = dx*dy*np.ones([N_x,N_y])
        
    # Current time of the simulation and step counter
    t_n = 0.0
    steps = 0

    csv_path = results_path + "csv_files/"
    figures_path = results_path + "figures/"

    rho_plot_path = figures_path+"rho-plot/"
    J_plot_path = figures_path+"J-plot/"
    A1_plot_path = figures_path+"A1-plot/"
    A2_plot_path = figures_path+"A2-plot/"
    psi_plot_path = figures_path+"phi-plot/"
    gauge_plot_path = figures_path+"gauge-plot/"
    gauss_plot_path = figures_path+"gauss-plot/"
    E_plot_path = figures_path+"E-plot/"
    B_plot_path = figures_path+"B-plot/"

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

    if (enable_plots):
        if(not os.path.exists(rho_plot_path)):
            os.makedirs(rho_plot_path)
    
        if(not os.path.exists(J_plot_path)):
            os.makedirs(J_plot_path)

        if(not os.path.exists(A1_plot_path)):
            os.makedirs(A1_plot_path)

        if(not os.path.exists(A2_plot_path)):
            os.makedirs(A2_plot_path)
    
        if(not os.path.exists(psi_plot_path)):
            os.makedirs(psi_plot_path)

        if(not os.path.exists(gauge_plot_path)):
            os.makedirs(gauge_plot_path)
            os.makedirs(gauge_plot_path+"/surface")
            os.makedirs(gauge_plot_path+"/slice")

        if(not os.path.exists(gauss_plot_path)):
            os.makedirs(gauss_plot_path)
            os.makedirs(gauss_plot_path+"/surface")
            os.makedirs(gauss_plot_path+"/slice")

        if(not os.path.exists(E_plot_path)):
            os.makedirs(E_plot_path)
    
        if(not os.path.exists(B_plot_path)):
            os.makedirs(B_plot_path)
    
    if (enable_csvs):

        if(not os.path.exists(electron_csv_path)):
            os.makedirs(electron_csv_path)

        if(not os.path.exists(rho_csv_path)):
            os.makedirs(rho_csv_path)
    
        if(not os.path.exists(J1_csv_path)):
            os.makedirs(J1_csv_path)
    
        if(not os.path.exists(J2_csv_path)):
            os.makedirs(J2_csv_path)

        if(not os.path.exists(A1_csv_path)):
            os.makedirs(A1_csv_path)

        if(not os.path.exists(A2_csv_path)):
            os.makedirs(A2_csv_path)
    
        if(not os.path.exists(psi_csv_path)):
            os.makedirs(psi_csv_path)

        if(not os.path.exists(gauge_csv_path)):
            os.makedirs(gauge_csv_path)

        if(not os.path.exists(gauss_csv_path)):
            os.makedirs(gauss_csv_path)

        if(not os.path.exists(E1_csv_path)):
            os.makedirs(E1_csv_path)

        if(not os.path.exists(E2_csv_path)):
            os.makedirs(E2_csv_path)
    
        if(not os.path.exists(B3_csv_path)):
            os.makedirs(B3_csv_path)


    N_elec_hist = np.zeros([N_steps])
    csv = ".csv"
    jpg = ".jpg"


    while(steps < N_steps):

        if (steps % plot_at == 0):
            print(steps)
        
        #---------------------------------------------------------------------
        # 1. Advance electron positions by dt using v^{n}
        #---------------------------------------------------------------------
         
        advance_particle_positions_2D(x1_elec_new, x2_elec_new,
                                      x1_elec_old, x2_elec_old,
                                      v1_elec_old, v2_elec_old, dt)

        #---------------------------------------------------------------------
        # 2. Compute the electron current density used for updating A
        #---------------------------------------------------------------------

        # Clear the contents of J prior to the mapping
        # This is done here b/c the J function does not reset the current
        # We do this so that it can be applied to any number of species
        N_elec_now = remove_particles(x1_elec_old[:N_elec_now], x2_elec_old[:N_elec_now],
                                      x1_elec_new[:N_elec_now], x2_elec_new[:N_elec_now],
                                      v1_elec_old[:N_elec_now], v2_elec_old[:N_elec_now],
                                      v1_elec_new[:N_elec_now], v2_elec_new[:N_elec_now],
                                      P1_elec_old[:N_elec_now], P2_elec_old[:N_elec_now],
                                      P1_elec_new[:N_elec_now], P2_elec_new[:N_elec_now],
                                      v1_elec_nm1[:N_elec_now], v2_elec_nm1[:N_elec_now], # Need the additional history here
                                      x[0], x[-1], y[0], y[-1])
        
        N_elec_hist[steps] = N_elec_now
        
        J_mesh[:,:,:] = 0.0
        
        # Map for electrons (ions are stationary)
        # Can try using the starred velocities here if we want
        map_J_to_mesh_2D2V(J_mesh[:,:,:], x, y, dx, dy,
                           x1_elec_new[:N_elec_now], x2_elec_new[:N_elec_now], 
                           v1_elec_old[:N_elec_now], v2_elec_old[:N_elec_now],
                           q_elec, cell_volumes, w_elec)
        
        # There should be no current along the boundary (tangential direction)
        J_mesh[1,0 ,:] = 0.0
        J_mesh[1,-1,:] = 0.0

        J_mesh[0,:,0 ] = 0.0
        J_mesh[0,:,-1] = 0.0

        # Compute components of div(J) using finite-differences
        compute_ddx_FD(ddx_J1, J_mesh[0,:,:], dx)
        compute_ddy_FD(ddy_J2, J_mesh[1,:,:], dy)
        
        #---------------------------------------------------------------------
        # 4. Using the new positions, map charge to the mesh to get rho^{n+1}
        #---------------------------------------------------------------------
        
        # Clear the contents of rho at time level n+1
        # prior to the mapping
        rho_ions[:,:] = 0.0
        rho_elec[:,:] = 0.0

        # Ions
        map_rho_to_mesh_2D(rho_ions[:,:], x, y, dx, dy,
                           x1_ions, x2_ions,
                           q_ions, cell_volumes, w_ions)
        
        # Electrons
        map_rho_to_mesh_2D(rho_elec[:,:], x, y, dx, dy,
                           x1_elec_new[:N_elec_now], x2_elec_new[:N_elec_now],
                           q_elec, cell_volumes, w_elec)
        
        rho_mesh[:,:] = rho_ions[:,:] + rho_elec[:,:]
        
        # There should be no charge along the boundary
        rho_mesh[0 ,:] = 0.0
        rho_mesh[-1,:] = 0.0

        rho_mesh[:,0 ] = 0.0
        rho_mesh[:,-1] = 0.0
        
        #---------------------------------------------------------------------
        # 5. Advance the psi and its derivatives by dt using BDF-1 
        #---------------------------------------------------------------------
        
        psi_src[:,:] = (1/sigma_1)*rho_mesh[:,:]
        
        # Charge density is at the new time level from step (3)
        # which is consistent with the BDF scheme
        BDF1_combined_dir_advance(psi, ddx_psi, ddy_psi, psi_src,
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # Wait to shuffle until the end, but we could do that here
        
        #---------------------------------------------------------------------
        # 5. Advance the A1 and A2 and their derivatives by dt using BDF-1
        #---------------------------------------------------------------------
        
        A1_src[:,:] = sigma_2*J_mesh[0,:,:]
        A2_src[:,:] = sigma_2*J_mesh[1,:,:]
        
        # A1 uses J1
        BDF1_combined_dir_advance(A1, ddx_A1, ddy_A1, A1_src[:,:],
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # A2 uses J2
        BDF1_combined_dir_advance(A2, ddx_A2, ddy_A2, A2_src[:,:],
                                  x, y, t_n, dx, dy, dt, kappa, beta_BDF)
        
        # Wait to shuffle until the end, but we could do that here
        
        #---------------------------------------------------------------------
        # 6. Momentum advance by dt
        #---------------------------------------------------------------------
        
        # Fields are taken implicitly and we use the "lagged" velocity
        #
        # This will give us new momenta and velocities for the next step
        improved_asym_euler_momentum_push_2D2P(P1_elec_new[:N_elec_now], P2_elec_new[:N_elec_now],
                                               v1_elec_new[:N_elec_now], v2_elec_new[:N_elec_now],
                                               x1_elec_new[:N_elec_now], x2_elec_new[:N_elec_now],
                                               P1_elec_old[:N_elec_now], P2_elec_old[:N_elec_now],
                                               v1_elec_old[:N_elec_now], v2_elec_old[:N_elec_now],
                                               v1_elec_nm1[:N_elec_now], v2_elec_nm1[:N_elec_now],
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
        
        if enable_csvs and (steps % plot_at == 0 or steps + 1 == N_steps):
            
            # step_pad = "{:0>6}".format(steps)
            step_pad = str(steps).zfill(6)

            particle_information = np.column_stack([x1_elec_old, x2_elec_old, v1_elec_old, v2_elec_old])

            np.savetxt(electron_csv_path+"electron"+step_pad+csv, particle_information, delimiter=",")
            np.savetxt(rho_csv_path+"rho"+step_pad+csv, rho_mesh, delimiter=",")
            np.savetxt(J1_csv_path + "J1_"+step_pad+csv, J_mesh[0,:,:], delimiter=",")
            np.savetxt(J2_csv_path + "J2_"+step_pad+csv, J_mesh[1,:,:], delimiter=",")
            np.savetxt(A1_csv_path + "A1_"+step_pad+csv, A1[-1,:,:], delimiter=",")
            np.savetxt(A2_csv_path + "A2_"+step_pad+csv, A2[-1,:,:], delimiter=",")
            np.savetxt(psi_csv_path + "phi_"+step_pad+csv, psi[-1,:,:], delimiter=",")
            np.savetxt(gauge_csv_path + "gauge_"+step_pad+csv,gauge_residual, delimiter=",")
            np.savetxt(gauss_csv_path + "gauss_"+step_pad+csv,gauss_law_residual, delimiter=",")
            np.savetxt(E1_csv_path + "E1_"+step_pad+csv, E1, delimiter=",")
            np.savetxt(E2_csv_path + "E2_"+step_pad+csv, E2, delimiter=",")
            np.savetxt(B3_csv_path + "B3_"+step_pad+csv, B3, delimiter=",")

        if enable_plots:

            # step_pad = "{:0>6}".format(steps)
            step_pad = str(steps).zfill(6)
            
            # Should also plot things at the final step as well
            if steps % plot_at == 0 or steps + 1 == N_steps:
                
                print("Finished with step:", steps,"\n")
                
                # Don't measure the charge at the redundant boundary points
                # print("Total charge:","{:.6e}".format(np.sum(cell_volumes[:-1,:-1]*rho_mesh[:-1,:-1])),"\n")
                # print("L2 error for the Gauge:","{:.6e}".format(gauge_error[steps]),"\n")
                # print("L2 error for Gauss' law:","{:.6e}".format(gauss_law_error[steps]),"\n")
                # print("Sum of the residual for Gauss' law:","{:.6e}".format(sum_gauss_law_residual[steps]),"\n")
                print("Time: ", dt*steps, " Number of electrons: ", N_elec_now, "\n")
                
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
                plt.savefig(rho_plot_path+"rho_"+str(step_pad)+jpg, bbox_inches="tight")                
                plt.close(fig="all")

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
                plt.savefig(J_plot_path+"J_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")
                
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
                plt.savefig(A1_plot_path+"A1_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")
                
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
                plt.savefig(A2_plot_path+"A2_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")
                
                # 3D Plot of psi, ddx_psi, and ddy_psi                 
                fig = plt.figure(figsize=plt.figaspect(1/3))
                ax = fig.add_subplot(1,3,1, projection='3d')
                surf = ax.plot_surface(X, Y, psi[-1,:,:], cmap = 'viridis')
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
                ax.tick_params(axis='x')
                ax.tick_params(axis='y')
                # ax.xaxis.offsetText.set_fontsize(32)
                # ax.yaxis.offsetText.set_fontsize(32)
                ax.set_xlim((x[0],x[-1]))
                ax.set_ylim((y[0],y[-1]))
                ax.set_title( r"$\phi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                fig.colorbar(surf, shrink=0.8, ax=ax, pad=0.1)

                ax = fig.add_subplot(1,3,2, projection='3d')
                surf = ax.plot_surface(X, Y, ddx_psi, cmap = 'viridis')
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
                ax.tick_params(axis='x')
                ax.tick_params(axis='y')
                # ax.xaxis.offsetText.set_fontsize(32)
                # ax.yaxis.offsetText.set_fontsize(32)
                ax.set_xlim((x[0],x[-1]))
                ax.set_ylim((y[0],y[-1]))
                ax.set_title( r"$\partial_{x} \phi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                fig.colorbar(surf, shrink=0.8, ax=ax, pad=0.1)

                ax = fig.add_subplot(1,3,3, projection='3d')
                surf = ax.plot_surface(X, Y, ddx_psi, cmap = 'viridis')
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$")
                ax.tick_params(axis='x')
                ax.tick_params(axis='y')
                # ax.xaxis.offsetText.set_fontsize(32)
                # ax.yaxis.offsetText.set_fontsize(32)
                ax.set_xlim((x[0],x[-1]))
                ax.set_ylim((y[0],y[-1]))

                ax.set_title( r"$\partial_{y} \phi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                fig.colorbar(surf, shrink=0.8, ax=ax, pad=0.1)

                plt.tight_layout()
                plt.savefig(psi_plot_path+"phi_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")
                # Plot of psi, ddx_psi, and ddy_psi
                # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28,8), sharex=False, sharey=True)
                
                # im = axes[0].pcolormesh(X, Y, psi[-1,:,:], cmap = 'viridis', shading='auto')
                # axes[0].set_xlabel(r"$x$", fontsize=32)
                # axes[0].set_ylabel(r"$y$", fontsize=32)
                # axes[0].tick_params(axis='x', labelsize=32, pad=10)
                # axes[0].tick_params(axis='y', labelsize=32, pad=10)
                # axes[0].xaxis.offsetText.set_fontsize(32)
                # axes[0].yaxis.offsetText.set_fontsize(32)
                # axes[0].set_xlim((x[0],x[-1]))
                # axes[0].set_ylim((y[0],y[-1]))
                # axes[0].set_title( r"$\psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                # cbar = plt.colorbar(im, ax=axes[0])
                # cbar.ax.tick_params(labelsize=32)
                # cbar.ax.yaxis.offsetText.set(size=32)
                
                # im = axes[1].pcolormesh(X, Y, ddx_psi, cmap = 'viridis', shading='auto')
                # axes[1].set_xlabel(r"$x$", fontsize=32)
                # #axes[1].set_ylabel(r"$y$", fontsize=32)
                # axes[1].tick_params(axis='x', labelsize=32, pad=10)
                # axes[1].tick_params(axis='y', labelsize=32, pad=10)
                # axes[1].xaxis.offsetText.set_fontsize(32)
                # axes[1].yaxis.offsetText.set_fontsize(32)
                # axes[1].set_xlim((x[0],x[-1]))
                # axes[1].set_ylim((y[0],y[-1]))
                # axes[1].set_title( r"$\partial_{x} \psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                # cbar = plt.colorbar(im, ax=axes[1])
                # cbar.ax.tick_params(labelsize=32)
                # cbar.ax.yaxis.offsetText.set(size=32)
                
                # im = axes[2].pcolormesh(X, Y, ddy_psi, cmap = 'viridis', shading='auto')
                # axes[2].set_xlabel(r"$x$", fontsize=32)
                # #axes[2].set_ylabel(r"$y$", fontsize=32)
                # axes[2].tick_params(axis='x', labelsize=32, pad=10)
                # axes[2].tick_params(axis='y', labelsize=32, pad=10)
                # axes[2].xaxis.offsetText.set_fontsize(32)
                # axes[2].yaxis.offsetText.set_fontsize(32)
                # axes[2].set_xlim((x[0],x[-1]))
                # axes[2].set_ylim((y[0],y[-1]))
                # axes[2].set_title( r"$\partial_{y} \psi$ at $t = $ " + "{:.4e}".format(t_n+dt), fontsize=28 )
                # cbar = plt.colorbar(im, ax=axes[2])
                # cbar.ax.tick_params(labelsize=32)
                # cbar.ax.yaxis.offsetText.set(size=32)
                
                # plt.tight_layout()
                # plt.savefig(psi_path+"psi_"+str(step_pad)+jpg, bbox_inches="tight")
                

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
                               
                plt.savefig(gauge_plot_path+"slice/"+"gauge_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")   
                    
                fig = plt.figure(figsize=(12,10))
                ax = fig.add_subplot(projection='3d')

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
                
                plt.savefig(gauge_plot_path+"surface/"+"gauge_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")
        
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
                               
                plt.savefig(gauss_plot_path+"slice/"+"gauss_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")

                fig = plt.figure(figsize=(12,10))
                ax = fig.add_subplot(projection='3d')

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
                
                plt.savefig(gauss_plot_path+"surface/"+"gauss_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")

                
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
                plt.savefig(E_plot_path+"E_"+str(step_pad)+jpg, bbox_inches="tight")
                plt.close(fig="all")

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
                
                plt.savefig(B_plot_path+"B3_"+str(step_pad)+jpg, bbox_inches="tight")
                
                # Close the figures
                plt.close(fig="all")

        # Step is now complete
        steps += 1
        t_n += dt
        
    # Stop the timer
    solver_end_time = time.time()

    total_time = solver_end_time - solver_start_time
    
    # Before exiting, we write the following data to files:
    # Particle data: x1,x2
    # Sources: rho, J
    # Fields: E, B, even though we don't use these to push particles
    
    # # Write the particle beam data to a file
    # np.savetxt('EBP-aem1-taylor-beam-particles.csv', 
    #            np.transpose((x1_elec_new[:N_elec_now], x2_elec_new[:N_elec_now])), 
    #            delimiter=',')
    
    # # Write the charge density and current density components to files
    # np.savetxt('EBP-aem1-taylor-rho.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), rho_mesh.flatten())), 
    #            delimiter=',', header='x, y, rho')
    
    # np.savetxt('EBP-aem1-taylor-J1.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), J_mesh[0,:,:].flatten())), 
    #            delimiter=',', header='x, y, J1')
    
    # np.savetxt('EBP-aem1-taylor-J2.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), J_mesh[1,:,:].flatten())), 
    #            delimiter=',', header='x, y, J2')
    
    # # Fields are next: E1, E2, B3
    # np.savetxt('EBP-aem1-taylor-E1.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), E1.flatten())), 
    #            delimiter=',', header='x, y, E1')
    
    # np.savetxt('EBP-aem1-taylor-E2.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), E2.flatten())), 
    #            delimiter=',', header='x, y, E2')
    
    # np.savetxt('EBP-aem1-taylor-B3.csv', 
    #            np.transpose((X.flatten(), Y.flatten(), B3.flatten())), 
    #            delimiter=',', header='x, y, B3')

    return total_time, gauge_error, gauss_law_error, sum_gauss_law_residual, N_elec_hist, v_elec_var_history