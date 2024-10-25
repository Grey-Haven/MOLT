% This script is agnostic, it accepts any starting parameters and runs them
% with a second order MOLT method, using the charge/current updates that
% are passed in, the type of gauge correction (if any), and the visualization
% method (which must have the same parameters regardless of if they're actually
% displayed), which can be shared by all.

% create_plots(x, x1_elec_hist(:,end-1), v1_elec_hist(:,end-1), 0, tag, CFL, vidObj);

steps = 0;
t_n = steps*dt;

while(steps < N_steps)

    x1_elec_old = x1_elec_hist(:,end-1);

    v1_elec_old = v1_elec_hist(:,end-1);

    v1_elec_nm1 = v1_elec_hist(:,end-2);

    P1_elec_old = P1_elec_hist(:,end-1);

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots(x, x1_elec_old, v1_elec_old, psi(:,end-1), rho_mesh, t_n, tag, CFL, vidObj);
        end
    end

    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------

    v1_star = 2*v1_elec_old - v1_elec_nm1;
    x1_elec_new = advance_particle_positions_1D(x1_elec_old, v1_star, dt);

    % Apply the particle boundary conditions
    % Need to include the shift function here
    x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 2. Scatter the source (rho)
    %---------------------------------------------------------------------
    rho_compute_vanilla_1D;

    %---------------------------------------------------------------------
    % 3.1 Update the scalar (phi)
    %---------------------------------------------------------------------

    psi_src(:) = (1/sigma_1)*rho_mesh;
    [psi, ddx_psi] = BDF1_combined_per_advance_1D(psi, ddx_psi, psi_src, x, t_n, dx, dt, kappa, beta_BDF1);


    %---------------------------------------------------------------------
    % 4. Momentum advance by dt
    %---------------------------------------------------------------------
    [v1_elec_new, P1_elec_new] = ...
        improved_asym_euler_momentum_push_1D1P(x1_elec_new, P1_elec_old, v1_elec_old, v1_elec_nm1, ...
                                                ddx_psi(:,end), A1(:,end), ddx_A1(:,end), ...
                                                x, dx, q_elec, r_elec, ...
                                                kappa, dt);

    %---------------------------------------------------------------------
    % 6. Prepare for the next time step by shuffling the time history data
    %---------------------------------------------------------------------
    x1_elec_hist(:,end) = x1_elec_new;
    
    v1_elec_hist(:,end) = v1_elec_new;
    
    P1_elec_hist(:,end) = P1_elec_new;

    % Shuffle the time history of the fields
    psi = shuffle_steps(psi);
    ddx_psi = shuffle_steps(ddx_psi);

    A1 = shuffle_steps(A1);
    ddx_A1 = shuffle_steps(ddx_A1);

    % Shuffle the time history of the particle data
    x1_elec_hist = shuffle_steps(x1_elec_hist);

    v1_elec_hist = shuffle_steps(v1_elec_hist);

    P1_elec_hist = shuffle_steps(P1_elec_hist);

    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

    E_hist(steps) = sqrt(dx*sum(ddx_psi(1:end-1).^2));
    % E_hist(steps) = sqrt(dx*sum(ddx_psi.^2));

end

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots(x, x1_elec_old, v1_elec_old, psi(:,end-1), rho_mesh, t_n, tag, CFL, vidObj);
    close(vidObj);
end

ts = 0:dt:(N_steps-1)*dt;
% semilogy(ts, E_hist);
% hold on;

writematrix([ts', E_hist]', results_path + "E_hist.csv");