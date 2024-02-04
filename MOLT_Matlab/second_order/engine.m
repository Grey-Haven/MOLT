% This script is agnostic, it accepts any starting parameters and runs them
% with a second order MOLT method, using the charge/current updates that
% are passed in, the type of gauge correction (if any), and the visualization
% method (which must have the same parameters regardless of if they're actually
% displayed), which can be shared by all.

steps = 0;

dt_half = dt / 2;
dt_full = dt;

dt = dt_half;
for i = 1:6
    full_step_forward_first_order;
    create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
end
dt = dt_full;


% Situation as it is:
% We have six steps 
%
% |     0     |    1/2    |     1     |    3/2    |     2     |     3/2     |
%      ALL         ALL         ALL         ALL         ALL          ALL
%
% We need to stagger
%
% |     0     |    1/2    |     1     |    3/2    |     2     |     3/2     |
%  {x,rho,phi}   {v,J,A}   {x,rho,phi}   {v,J,A}   {x,rho,phi}    {v,J,A}   

x1_elec_0 = x1_elec_hist(:,end-5);
x1_elec_1 = x1_elec_hist(:,end-3);
x1_elec_2 = x1_elec_hist(:,end-1);
x2_elec_0 = x2_elec_hist(:,end-5);
x2_elec_1 = x2_elec_hist(:,end-3);
x2_elec_2 = x2_elec_hist(:,end-1);

rho_mesh_0 = rho_mesh(:,:,end-5);
rho_mesh_1 = rho_mesh(:,:,end-3);
rho_mesh_2 = rho_mesh(:,:,end-1);

psi_0 = psi(:,:,end-5);
psi_1 = psi(:,:,end-3);
psi_2 = psi(:,:,end-1);


v1_elec_1_2 = v1_elec_hist(:,end-4);
v1_elec_3_2 = v1_elec_hist(:,end-2);
v1_elec_5_2 = v1_elec_hist(:,end  );
v2_elec_1_2 = v2_elec_hist(:,end-4);
v2_elec_3_2 = v2_elec_hist(:,end-2);
v2_elec_5_2 = v2_elec_hist(:,end  );

J1_1_2 = J1_mesh(:,:,end-4);
J1_3_2 = J1_mesh(:,:,end-2);
J1_5_2 = J1_mesh(:,:,end  );
J2_1_2 = J2_mesh(:,:,end-4);
J2_3_2 = J2_mesh(:,:,end-2);
J2_5_2 = J2_mesh(:,:,end  );

A1_1_2 = A1(:,:,end-4);
A1_3_2 = A1(:,:,end-2);
A1_5_2 = A1(:,:,end  );
A2_1_2 = A2(:,:,end-4);
A2_3_2 = A2(:,:,end-2);
A2_5_2 = A2(:,:,end  );


x1_elec_hist(:,end-2) = x1_elec_0;
x1_elec_hist(:,end-1) = x1_elec_1;
x1_elec_hist(:,end  ) = x1_elec_2;
x2_elec_hist(:,end-2) = x2_elec_0;
x2_elec_hist(:,end-1) = x2_elec_1;
x2_elec_hist(:,end  ) = x2_elec_2;

rho_mesh(:,:,end-2) = rho_mesh_0;
rho_mesh(:,:,end-1) = rho_mesh_1;
rho_mesh(:,:,end  ) = rho_mesh_2;

psi(:,:,end-2) = psi_0;
psi(:,:,end-1) = psi_1;
psi(:,:,end  ) = psi_2;


v1_elec_hist(:,end-2) = v1_elec_1_2;
v1_elec_hist(:,end-1) = v1_elec_3_2;
v1_elec_hist(:,end  ) = v1_elec_5_2;
v2_elec_hist(:,end-2) = v2_elec_1_2;
v2_elec_hist(:,end-1) = v2_elec_3_2;
v2_elec_hist(:,end  ) = v2_elec_5_2;

J1_mesh(:,:,end-2) = J1_1_2;
J1_mesh(:,:,end-1) = J1_3_2;
J1_mesh(:,:,end  ) = J1_5_2;
J2_mesh(:,:,end-2) = J2_1_2;
J2_mesh(:,:,end-1) = J2_3_2;
J2_mesh(:,:,end  ) = J2_5_2;

A1(:,:,end-2) = A1_1_2;
A1(:,:,end-1) = A1_3_2;
A1(:,:,end  ) = A1_5_2;
A2(:,:,end-2) = A2_1_2;
A2(:,:,end-1) = A2_3_2;
A2(:,:,end  ) = A2_5_2;


create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
             gauge_residual, gauss_residual, ...
             x1_elec_new, x2_elec_new, t_n, ...
             update_method_title, tag, vidObj);


while(steps < N_steps)

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
                         gauge_residual, gauss_residual, ...
                         x1_elec_new, x2_elec_new, t_n, ...
                         update_method_title, tag, vidObj);
        end
    end

    x1_elec_old = x1_elec_hist(:,end-1);
    x2_elec_old = x2_elec_hist(:,end-1);

    v1_elec_old = v1_elec_hist(:,end-1);
    v2_elec_old = v2_elec_hist(:,end-1);

    v1_elec_nm1 = v1_elec_hist(:,end-2);
    v2_elec_nm1 = v2_elec_hist(:,end-2);

    P1_elec_old = P1_elec_hist(:,end-1);
    P2_elec_old = P2_elec_hist(:,end-1);

    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------

    v1_star = 2*v1_elec_old - v1_elec_nm1;
    v2_star = 2*v2_elec_old - v2_elec_nm1;
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                               v1_star, v2_star, dt);

    
    % Apply the particle boundary conditions
    % Need to include the shift function here
    x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
    x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 2. Compute the electron current density used for updating A
    %    Compute also the charge density used for updating psi
    %---------------------------------------------------------------------

    if J_rho_update_method == J_rho_update_method_vanilla
        J_rho_update_vanilla;
    elseif J_rho_update_method == J_rho_update_method_FFT
        x1_elec_store = x1_elec_new;
        x2_elec_store = x2_elec_new;
        x1_elec_new = (x1_elec_new + x1_elec_old)/2;
        x2_elec_new = (x2_elec_new + x2_elec_old)/2;
        J_rho_update_fft;
        x1_elec_new = x1_elec_store;
        x2_elec_new = x2_elec_store;
    elseif J_rho_update_method == J_rho_update_method_FD6
        J_rho_update_FD6;
    else
        ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
        throw(ME);
    end
    
    %---------------------------------------------------------------------
    % 3.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*(rho_mesh(:,:,end) + rho_mesh(:,:,end-2));
    A1_src(:,:)  =     sigma_2*(J1_mesh(:,:,end)  + J1_mesh(:,:,end-2) );
    A2_src(:,:)  =     sigma_2*(J2_mesh(:,:,end)  + J2_mesh(:,:,end-2) );

    %---------------------------------------------------------------------
    % 3.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
    if waves_update_method == waves_update_method_vanilla
        update_waves_vanilla_second_order;
    elseif waves_update_method == waves_update_method_FFT
        update_waves_hybrid_FFT_second_order;
    elseif waves_update_method == waves_update_method_FD6
        update_waves_hybrid_FD6_second_order;
    elseif waves_update_method == waves_update_method_poisson_phi
        update_waves_poisson_phi_second_order;
    elseif waves_update_method == waves_update_method_pure_FFT
        update_waves_pure_FFT_second_order;
    else
        ME = MException('WaveException','Wave Method ' + wave_update_method + " not an option");
        throw(ME);
    end
    
    %---------------------------------------------------------------------
    % 3.3 Correct gauge error (optional)
    %---------------------------------------------------------------------
    if gauge_correction == gauge_correction_none
        % Nothing
    elseif gauge_correction == gauge_correction_FFT
        gauge_correction_FFT_deriv;
    elseif gauge_correction == gauge_correction_FD6
        gauge_correction_FD6_deriv;
    else
        ME = MException('GaugeCorrectionException','Gauge Correction Method ' + gauge_correction + " not an option");
        throw(ME);
    end
    

    %---------------------------------------------------------------------
    % 4. Momentum advance by dt
    %---------------------------------------------------------------------
    
    % Fields are taken implicitly and we use the "lagged" velocity
    %
    % This will give us new momenta and velocities for the next step
    ddx_psi_ave = (ddx_psi(:,:,end) + ddx_psi(:,:,end-1))/2;
    ddy_psi_ave = (ddy_psi(:,:,end) + ddy_psi(:,:,end-1))/2;
    [v1_elec_new, v2_elec_new, P1_elec_new, P2_elec_new] = ...
    improved_asym_euler_momentum_push_2D2P_implicit(x1_elec_new, x2_elec_new, ...
                                                    P1_elec_old, P2_elec_old, ...
                                                    v1_elec_old, v2_elec_old, ...
                                                    v1_elec_nm1, v2_elec_nm1, ...
                                                    ddx_psi_ave, ddy_psi_ave, ...
                                                    A1(:,:,end), ddx_A1(:,:,end), ddy_A1(:,:,end), ...
                                                    A2(:,:,end), ddx_A2(:,:,end), ddy_A2(:,:,end), ...
                                                    x, y, dx, dy, q_elec, r_elec, ...
                                                    kappa, dt);
                                       
    %---------------------------------------------------------------------
    % 5. Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) ) / dt;

%     ddx_A1_ave = (ddx_A1(:,:,end) + ddx_A1(:,:,end-1)) / 2;
%     ddy_A2_ave = (ddy_A2(:,:,end) + ddy_A2(:,:,end-1)) / 2;
    
    % Compute the residual in the Lorenz gauge 
%     gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1_ave(:,:,end) + ddy_A2_ave(:,:,end);
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));

    rho_hist(steps+1) = sum(sum(rho_elec(1:end-1,1:end-1)));

    compute_gauss_residual_second_order;

    
    %---------------------------------------------------------------------
    % 6. Prepare for the next time step by shuffling the time history data
    %---------------------------------------------------------------------
    x1_elec_hist(:,end) = x1_elec_new;
    x2_elec_hist(:,end) = x2_elec_new;
    
    v1_elec_hist(:,end) = v1_elec_new;
    v2_elec_hist(:,end) = v2_elec_new;
    
    P1_elec_hist(:,end) = P1_elec_new;
    P2_elec_hist(:,end) = P2_elec_new;

    % Shuffle the time history of the fields
    psi = shuffle_steps(psi);
    ddx_psi = shuffle_steps(ddx_psi);
    ddy_psi = shuffle_steps(ddy_psi);

    A1 = shuffle_steps(A1);
    ddx_A1 = shuffle_steps(ddx_A1);
    ddy_A1 = shuffle_steps(ddy_A1);

    A2 = shuffle_steps(A2);
    ddx_A2 = shuffle_steps(ddx_A2);
    ddy_A2 = shuffle_steps(ddy_A2);

    rho_mesh = shuffle_steps(rho_mesh);
    J1_mesh = shuffle_steps(J1_mesh);
    J2_mesh = shuffle_steps(J2_mesh);
    
    % Shuffle the time history of the particle data
    v1_elec_hist = shuffle_steps(v1_elec_hist);
    v2_elec_hist = shuffle_steps(v2_elec_hist);
    x1_elec_hist = shuffle_steps(x1_elec_hist);
    x2_elec_hist = shuffle_steps(x2_elec_hist);
    P1_elec_hist = shuffle_steps(P1_elec_hist);
    P2_elec_hist = shuffle_steps(P2_elec_hist);

    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

end

ts = 0:dt:(N_steps-1)*dt;

gauge_error_array = zeros(length(ts),3);
gauge_error_array(:,1) = ts;
gauge_error_array(:,2) = gauge_error_L2;
gauge_error_array(:,3) = gauge_error_inf;

gauss_error_array = zeros(length(ts),7);
gauss_error_array(:,1) = ts;
gauss_error_array(:,2) = gauss_law_potential_err_L2;
gauss_error_array(:,3) = gauss_law_potential_err_inf;
gauss_error_array(:,4) = gauss_law_gauge_err_L2;
gauss_error_array(:,5) = gauss_law_gauge_err_inf;
gauss_error_array(:,6) = gauss_law_field_err_L2;
gauss_error_array(:,7) = gauss_law_field_err_inf;

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
    close(vidObj);
end

writematrix(gauge_error_array, csvPath + "gauge_error.csv");
writematrix(gauss_error_array, csvPath + "gauss_error.csv");