%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle solver for the 2D-2P heating test that uses the asymmetrical Euler method for particles
% and the MOLT field solvers.
%
% Note that this problem starts out as charge neutral and with a net zero current. Therefore, the
% fields are taken to be zero initially.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while(steps < N_steps)

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots_heating;
            disp(tag + " " + num2str((t_n / T_final) * 100) + "% complete");
        else
            % waitbar((steps / N_steps), f, tag)
            disp(tag + " " + num2str((t_n / T_final) * 100) + "% complete");
        end
    end
    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------

    v1_star = 2*v1_elec_old - v1_elec_nm1;
    v2_star = 2*v2_elec_old - v2_elec_nm1;
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_new, x2_elec_new, ...
                                                               x1_elec_old, x2_elec_old, ...
                                                               v1_star, v2_star, dt);
%                                                                v1_elec_new, v2_elec_new, dt);

    
    % Apply the particle boundary conditions
    % Need to include the shift function here
    x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
    x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);

    %---------------------------------------------------------------------
    % 2. Compute the electron current density used for updating A
    %    Compute also the charge density used for updating psi
    %---------------------------------------------------------------------

    if J_rho_update_method == J_rho_update_method_vanilla
        J_rho_update_vanilla;
    elseif J_rho_update_method == J_rho_update_method_FFT
        J_rho_update_fft;
    elseif J_rho_update_method == J_rho_update_method_FD6
        J_rho_update_FD6;
    else
        ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
        throw(ME);
    end

    %---------------------------------------------------------------------
    % 3. Update the vector and scalar equations (wave solve)
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 3.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:);
    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    %---------------------------------------------------------------------
    % 3.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
    if waves_update_method == waves_update_method_vanilla
        update_waves;
    elseif waves_update_method == waves_update_method_FFT
        update_waves_hybrid_FFT;
    elseif waves_update_method == waves_update_method_FD6
        update_waves_hybrid_FD6
    elseif waves_update_method == waves_update_method_poisson_phi
        update_waves_poisson_phi;
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
    [v1_elec_new, v2_elec_new, P1_elec_new, P2_elec_new] = ...
    improved_asym_euler_momentum_push_2D2P(x1_elec_new, x2_elec_new, ...
                                           P1_elec_old, P2_elec_old, ...
                                           v1_elec_old, v2_elec_old, ...
                                           v1_elec_nm1, v2_elec_nm1, ...
                                           ddx_psi(:,:,end), ddy_psi(:,:,end), ...
                                           A1(:,:,end), ddx_A1(:,:,end), ddy_A1(:,:,end), ...
                                           A2(:,:,end), ddx_A2(:,:,end), ddy_A2(:,:,end), ...
                                           x, y, dx, dy, q_elec, r_elec, ...
                                           kappa, dt);


    %---------------------------------------------------------------------
    % 5. Diagnostics and Storage
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 5.1 Compute the errors in the Lorenz gauge
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite difference
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) )/dt;
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));
    
    %---------------------------------------------------------------------
    % 5.2 Compute the error in Gauss' Law
    %---------------------------------------------------------------------
    compute_gauss_residual;

    %---------------------------------------------------------------------
    % 5.3 Compute the electron variance (will be used to compute temp)
    %---------------------------------------------------------------------
    var_v1 = var(v1_elec_new);
    var_v2 = var(v2_elec_new);
    v_elec_var_hist(steps+1) = ( 0.5*(var_v1 + var_v2) );

    %-----------------------------------------------------------------------
    % 5.4 Store total charge
    %-----------------------------------------------------------------------
    rho_hist(steps+1) = sum(sum(rho_mesh(1:end-1,1:end-1)));

    %-----------------------------------------------------------------------
    % 5.5 Store magnitudes of E and B fields
    %-----------------------------------------------------------------------
    Ex_L2_hist(steps+1) = get_L_2_error(E1,zeros(size(E1)),dx*dy);
    Ey_L2_hist(steps+1) = get_L_2_error(E2,zeros(size(E2)),dx*dy);
    Bz_L2_hist(steps+1) = get_L_2_error(B3,zeros(size(B3)),dx*dy);

    %-----------------------------------------------------------------------
    % 5.6 Measure the total mass and energy of the system (ions + electrons)
    %-----------------------------------------------------------------------

    % Ions are stationary, so their total mass will not change
%     total_mass_elec = get_total_mass_species(rho_elec, cell_volumes, q_elec, r_elec);
%     
%     total_energy_ions = get_total_energy(psi(:,:,end), A1(:,:,end), A2(:,:,end), ...
%                                            x1_ions, x2_ions, ...
%                                            P1_ions, P2_ions, ...
%                                            x, y, q_ions, w_ions*r_ions, kappa);
%     
%     total_energy_elec = get_total_energy(psi(:,:,end), A1(:,:,end), A2(:,:,end), ...
%                                          x1_elec_new, x2_elec_new, ...
%                                          P1_elec_new, P2_elec_new, ...
%                                          x, y, q_elec, w_elec*r_elec, kappa);
%     
%     % Combine the species information
%     total_mass(steps+1) = total_mass_ions + total_mass_elec;
%     total_energy(steps+1) = total_energy_ions + total_energy_elec;

    
    %---------------------------------------------------------------------
    % 6. Prepare for the next time step by shuffling the time history data
    %---------------------------------------------------------------------
    
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
    
    % Shuffle the time history of the particle data
    v1_elec_nm1(:) = v1_elec_old(:);
    v2_elec_nm1(:) = v2_elec_old(:);
    
    x1_elec_old(:) = x1_elec_new(:);
    x2_elec_old(:) = x2_elec_new(:);
    
    v1_elec_old(:) = v1_elec_new(:);
    v2_elec_old(:) = v2_elec_new(:);
    
    P1_elec_old(:) = P1_elec_new(:);
    P2_elec_old(:) = P2_elec_new(:);

    % u_prev(:,:) = u_star(:,:);

%     if (mod(steps,50) == 0)
% 
%         temp_hist = ((M_electron * V^2) / k_B) * v_elec_var_hist;
%         
%         ts = 0:dt:steps*dt;
%         gauge_error_array = zeros(length(ts),3);
%         gauge_error_array(:,1) = ts;
%         gauge_error_array(:,2) = gauge_error_L2(1:steps+1);
%         gauge_error_array(:,3) = gauge_error_inf(1:steps+1);
%         
%         B3_L2_array = zeros(length(ts),2);
%         B3_L2_array(:,1) = ts;
%         B3_L2_array(:,2) = Bz_L2_hist(1:steps+1);
%         
%         E1_L2_array = zeros(length(ts),2);
%         E1_L2_array(:,1) = ts;
%         E1_L2_array(:,2) = Ex_L2_hist(1:steps+1);
%         
%         E2_L2_array = zeros(length(ts),2);
%         E2_L2_array(:,1) = ts;
%         E2_L2_array(:,2) = Ey_L2_hist(1:steps+1);
%         
%         rho_hist_array = zeros(length(ts),2);
%         rho_hist_array(:,1) = ts;
%         rho_hist_array(:,2) = rho_hist(1:steps+1);
%         
%         temp_hist_array = zeros(length(ts),2);
%         temp_hist_array(:,1) = ts;
%         temp_hist_array(:,2) = temp_hist(1:steps+1);
% 
%         writematrix(gauge_error_array,csvPath + "gauge_error" + tag + ".csv");
%         writematrix(temp_hist_array,csvPath + "temp_hist" + tag + ".csv");
%         writematrix(B3_L2_array,csvPath + "B3_magnitude" + tag + ".csv");
%         writematrix(E1_L2_array,csvPath + "E1_magnitude" + tag + ".csv");
%         writematrix(E2_L2_array,csvPath + "E2_magnitude" + tag + ".csv");
%         writematrix(rho_hist_array,csvPath + "rho_hist" + tag + ".csv");
% 
%     end

    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

end

temp_hist = ((M_electron * V^2) / k_B) * v_elec_var_hist;

ts = 0:dt:(N_steps-1)*dt;

gauge_error_array = zeros(length(ts),3);
gauge_error_array(:,1) = ts;
gauge_error_array(:,2) = gauge_error_L2;
gauge_error_array(:,3) = gauge_error_inf;

gauss_error_array = zeros(length(ts),3);
gauss_error_array(:,1) = ts;
gauss_error_array(:,2) = gauss_law_potential_err_L2;
gauss_error_array(:,3) = gauss_law_potential_err_inf;
gauss_error_array(:,4) = gauss_law_gauge_err_L2;
gauss_error_array(:,5) = gauss_law_gauge_err_inf;
gauss_error_array(:,6) = gauss_law_field_err_L2;
gauss_error_array(:,7) = gauss_law_field_err_inf;

B3_L2_array = zeros(length(ts),2);
B3_L2_array(:,1) = ts;
B3_L2_array(:,2) = Bz_L2_hist;

E1_L2_array = zeros(length(ts),2);
E1_L2_array(:,1) = ts;
E1_L2_array(:,2) = Ex_L2_hist;

E2_L2_array = zeros(length(ts),2);
E2_L2_array(:,1) = ts;
E2_L2_array(:,2) = Ey_L2_hist;

rho_hist_array = zeros(length(ts),2);
rho_hist_array(:,1) = ts;
rho_hist_array(:,2) = rho_hist;

temp_hist_array = zeros(length(ts),2);
temp_hist_array(:,1) = ts;
temp_hist_array(:,2) = temp_hist;

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots_heating;
    close(vidObj);
else
    close(f);
end

writematrix(gauge_error_array,csvPath + "gauge_error_" + tag + ".csv");
writematrix(gauss_error_array,csvPath + "gauss_error_" + tag + ".csv");
writematrix(temp_hist_array,csvPath + "temp_hist_" + tag + ".csv");
writematrix(B3_L2_array,csvPath + "B3_magnitude_" + tag + ".csv");
writematrix(E1_L2_array,csvPath + "E1_magnitude_" + tag + ".csv");
writematrix(E2_L2_array,csvPath + "E2_magnitude_" + tag + ".csv");
writematrix(rho_hist_array,csvPath + "rho_hist_" + tag + ".csv");