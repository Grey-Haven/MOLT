% This script is agnostic, it accepts any starting parameters and runs them
% with a second order MOLT method, using the charge/current updates that
% are passed in, the type of gauge correction (if any), and the visualization
% method (which must have the same parameters regardless of if they're actually
% displayed), which can be shared by all.

% Rev the engine a little
% fixed_point; % vroom

% create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
%              gauge_residual, gauss_residual, ...
%              x1_elec_new, x2_elec_new, t_n, ...
%              update_method_title, tag, vidObj);

steps = 0;

while(steps < N_steps)

    x1_elec_old = x1_elec_hist(:,end-1);
    x2_elec_old = x2_elec_hist(:,end-1);

    v1_elec_old = v1_elec_hist(:,end-1);
    v2_elec_old = v2_elec_hist(:,end-1);

    v1_elec_nm1 = v1_elec_hist(:,end-2);
    v2_elec_nm1 = v2_elec_hist(:,end-2);

    P1_elec_old = P1_elec_hist(:,end-1);
    P2_elec_old = P2_elec_hist(:,end-1);

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots(x, y, psi, A1, A2, ...
                         rho_mesh(:,:,end), J1_mesh(:,:,end), J2_mesh(:,:,end), ...
                         gauge_residual, gauss_residual, ...
                         x1_elec_old, x2_elec_old, t_n, ...
                         update_method_title, tag, vidObj);
        end
    end

    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------

    v1_star = 2*v1_elec_old - v1_elec_nm1;
    v2_star = 2*v2_elec_old - v2_elec_nm1;
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                               v1_star, v2_star, dt/2);

    
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
        J_rho_update_fft;
    elseif J_rho_update_method == J_rho_update_method_DIRK2
        J_rho_update_DIRK2;
    elseif J_rho_update_method == J_rho_update_method_FD2
        J_rho_update_FD2;
    elseif J_rho_update_method == J_rho_update_method_FD4
        J_rho_update_FD4;
    elseif J_rho_update_method == J_rho_update_method_FD6
        J_rho_update_FD6;
    else
        ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
        throw(ME);
    end

    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                               v1_star, v2_star, dt);

    
    % Apply the particle boundary conditions
    % Need to include the shift function here
    x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
    x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);
    
    %---------------------------------------------------------------------
    % 3.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src_hist(:,:,1) = psi_src;
    A1_src_hist(:,:,1)  = A1_src;
    A2_src_hist(:,:,1)  = A2_src;

    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:,end);
    A1_src(:,:)  =     sigma_2*J1_mesh(:,:,end);
    A2_src(:,:)  =     sigma_2*J2_mesh(:,:,end);

    psi_src_hist(:,:,2) = psi_src;
    A1_src_hist(:,:,2)  = A1_src;
    A2_src_hist(:,:,2)  = A2_src;

    %---------------------------------------------------------------------
    % 3.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
    if waves_update_method == waves_update_method_vanilla
        update_waves_vanilla_second_order;
    elseif waves_update_method == waves_update_method_FFT
        update_waves_hybrid_FFT_second_order;
    elseif waves_update_method == waves_update_method_DIRK2
        update_waves_hybrid_DIRK2;
    elseif waves_update_method == waves_update_method_FD2
        update_waves_hybrid_FD2_second_order;
    elseif waves_update_method == waves_update_method_FD4
        update_waves_hybrid_FD4_second_order;
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
%     ddx_psi_ave = (ddx_psi(:,:,end) + ddx_psi(:,:,end-1))/2;
%     ddy_psi_ave = (ddy_psi(:,:,end) + ddy_psi(:,:,end-1))/2;
    [v1_elec_new, v2_elec_new, P1_elec_new, P2_elec_new] = ...
    improved_asym_euler_momentum_push_2D2P_implicit(x1_elec_new, x2_elec_new, ...
                                                    P1_elec_old, P2_elec_old, ...
                                                    v1_elec_old, v2_elec_old, ...
                                                    v1_elec_nm1, v2_elec_nm1, ...
                                                    ddx_psi(:,:,end), ddy_psi(:,:,end), ...
                                                    A1(:,:,end), ddx_A1(:,:,end), ddy_A1(:,:,end), ...
                                                    A2(:,:,end), ddx_A2(:,:,end), ddy_A2(:,:,end), ...
                                                    x, y, dx, dy, q_elec, r_elec, ...
                                                    kappa, dt);

    %---------------------------------------------------------------------
    % 5. Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------

    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) ) / dt;

    b1 = 1/2;
    b2 = 1/2;

    c1 = 1/4;
    c2 = 3/4;

    div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
    div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);

    div_A_1 = (1-c1)*div_A_prev + c1*div_A_curr;
    div_A_2 = (1-c2)*div_A_prev + c2*div_A_curr;

    phi_1 = -div_A_1;
    phi_2 = -div_A_2;

%     ddx_A1_ave = (ddx_A1(:,:,end) + ddx_A1(:,:,end-1)) / 2;
%     ddy_A2_ave = (ddy_A2(:,:,end) + ddy_A2(:,:,end-1)) / 2;

    % Compute the residual in the Lorenz gauge 
%     gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1_ave(:,:,end) + ddy_A2_ave(:,:,end);
    % gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    % gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    % gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) - (b1*phi_1 + b2*phi_2);
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi - (b1*phi_1 + b2*phi_2);

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
    x1_elec_hist = shuffle_steps(x1_elec_hist);
    x2_elec_hist = shuffle_steps(x2_elec_hist);
    
    v1_elec_hist = shuffle_steps(v1_elec_hist);
    v2_elec_hist = shuffle_steps(v2_elec_hist);
    
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
    create_plots(x, y, psi, A1, A2, ...
                 rho_mesh(:,:,end), J1_mesh(:,:,end), J2_mesh(:,:,end), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
    close(vidObj);
end

writematrix(gauge_error_array, csvPath + "gauge_error.csv");
writematrix(gauss_error_array, csvPath + "gauss_error.csv");

figure;
x0=200;
y0=100;
width = 2400;
height = 1200;
set(gcf,'position',[x0,y0,width,height]);

subplot(1,2,1);
semilogy(ts,gauss_error_array(:,2))
hold on
semilogy(ts,gauss_error_array(:,4))
semilogy(ts,gauss_error_array(:,6))
hold off;

potential_label = "$\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta t^2} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$";
gauge_label = "$-\frac{\nabla\cdot\textbf{A}^{n+1} - \nabla\cdot\textbf{A}^{n}}{\Delta t} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$";
field_label = "$\nabla\cdot\textbf{E}^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$";
legend([potential_label,gauge_label,field_label],'FontSize',24,'interpreter','latex','location','east');
title("Gauss' Law",'FontSize',32);

subplot(1,2,2);
plot(ts,gauge_error_L2)
title("Gauge Error",'FontSize',32);

% quarter_len = floor(length(ts) / 4);
% 
% axes('Position',[.7 .6 .2 .2]);
% plot(ts(quarter_len:end),gauge_error_L2(quarter_len:end));

sgtitle(update_method_title + " " + tag,'FontSize',48);

saveas(gcf,figPath + "residuals.jpg");