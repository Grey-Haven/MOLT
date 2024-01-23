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
                                                               v1_elec_old, v2_elec_old, dt);
                                                               % v1_star, v2_star, dt);
%                                                                v1_elec_new, v2_elec_new, dt);

    
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
    elseif J_rho_update_method == J_rho_update_method_FD6
        J_rho_update_FD6;
    else
        ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
        throw(ME);
    end
%     J_rho_update_FD6;
    % J_rho_update_fft_iterative;    
    
    %---------------------------------------------------------------------
    % 5.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:);
    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    %---------------------------------------------------------------------
    % 5.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
    if waves_update_method == waves_update_method_vanilla
        update_waves;
    elseif waves_update_method == waves_update_method_FFT
        update_waves_hybrid_FFT;
    elseif waves_update_method == waves_update_method_FD6
        update_waves_hybrid_FD6
    else
        ME = MException('WaveException','Wave Method ' + wave_update_method + " not an option");
        throw(ME);
    end
    
    %---------------------------------------------------------------------
    % 5.5 Correct gauge error
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
    % 6. Momentum advance by dt
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
    % 7. Diagnostics and Storage
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 7.1 Compute the time derivatives of the potentials
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) )/dt;
    
    % Compute the ddt_A with backwards finite-differences
    ddt_A1(:,:) = ( A1(:,:,end) - A1(:,:,end-1) )/dt;
    ddt_A2(:,:) = ( A2(:,:,end) - A2(:,:,end-1) )/dt;

    %---------------------------------------------------------------------
    % 7.2 Compute the E and B fields
    %---------------------------------------------------------------------

    % Compute E = -grad(psi) - ddt_A
    % For ddt A, we use backward finite-differences
    % Note, E3 is not used in the particle update so we don't need ddt_A3
    E1(:,:) = -ddx_psi(:,:,end) - ddt_A1(:,:);
    E2(:,:) = -ddy_psi(:,:,end) - ddt_A2(:,:);
        
    % Compute B = curl(A)
    B3 = ddx_A2(:,:,end) - ddy_A1(:,:,end);

    %---------------------------------------------------------------------
    % 7.3 Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));
    
    var_vx = var(v1_elec_new);
    var_vy = var(v2_elec_new);
    var_v = (var_vx + var_vy)/2;
    v_elec_var_hist(steps+1) = var_v;

    %-----------------------------------------------------------------------
    % 7.4 Measure the total mass and energy of the system (ions + electrons)
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
    % 7.5 Compute the error in Gauss' Law
    %---------------------------------------------------------------------

    % Compute Gauss' law div(E) - rho to check the involution
    if waves_update_method == waves_update_method_vanilla
        ddx_E1 = compute_ddx_FD(E1, dx);
        ddy_E2 = compute_ddy_FD(E2, dy);
    elseif waves_update_method == waves_update_method_FFT
        ddx_E1 = compute_ddx_FFT(E1, kx_deriv_1);
        ddy_E2 = compute_ddy_FFT(E2, ky_deriv_1);
    elseif waves_update_method == waves_update_method_FD6
        ME = MException('WaveException',"FD6 Derivative not implemented yet.");
        throw(ME);
        % ddx_E1 = compute_ddx_FD6(E1, dx);
        % ddy_E2 = compute_ddy_FD6(E2, dy);
    else
        ME = MException('WaveException','Wave Method ' + wave_update_method + " not an option");
        throw(ME);
    end

    gauss_law_residual(:,:) = ddx_E1(:,:) + ddy_E2(:,:) - psi_src(:,:);
    
    gauss_law_error_L2(steps+1) = get_L_2_error(gauss_law_residual(:,:), ...
                                           zeros(size(gauss_law_residual(:,:))), ...
                                           dx*dy);
    gauss_law_error_inf(steps+1) = max(max(abs(gauss_law_residual(:,:))));
    
    % Now we measure the sum of the residual in Gauss' law (avoiding the boundary)
    sum_gauss_law_residual(steps+1) = sum(sum(gauss_law_residual(:,:)));

%     div_A_curr = ddx_A1(:,:,end) + ddy_A2(:,:,end);
%     div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
%     ddt_div_A = (div_A_curr - div_A_prev)/dt;
% 
%     ddt2_phi = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2))/(kappa^2*dt^2);
%     gauss_law_potential_form(steps+1) = get_L_2_error(ddt_div_A + ddt2_phi, ...
%                                              zeros(size(gauss_law_residual(:,:))), ...
%                                              dx*dy);
% 
%     laplacian_phi = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);
%     LHS = ddt2_phi - laplacian_phi;
%     res = LHS - rho_mesh / sigma_1;
%     divE = - laplacian_phi - ddt_div_A;
%     gauss_law_alt_residual = divE - rho_mesh;
%     gauss_law_alt_err = get_L_2_error(gauss_law_alt_residual, ...
%                                       zeros(size(gauss_law_residual(:,:))), ...
%                                       dx*dy);
%     subplot(1,3,1);
%     surf(x,y,LHS);
%     title("$\frac{1}{\kappa^2}\frac{\partial^2 \phi}{\partial t^2} - \Delta \phi$",'interpreter','latex','FontSize',24);
%     subplot(1,3,2);
%     surf(x,y,rho_mesh/sigma_1);
%     title("$\frac{\rho}{\sigma_1}$",'interpreter','latex','FontSize',24);
%     subplot(1,3,3);
%     surf(x,y,res);
%     title("$\left(\frac{1}{\kappa^2}\frac{\partial^2 \phi}{\partial t^2} - \Delta \phi\right) - \frac{\rho}{\sigma_1}$",'interpreter','latex','FontSize',24);
%     drawnow;

    
    %---------------------------------------------------------------------
    % 8. Prepare for the next time step by shuffling the time history data
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
    
    % % Measure the variance of the electron velocity distribution
    % and store for later use
    %
    % Note that we average the variance here so we don't need an
    % extra factor of two outside of this function
    var_v1 = var(v1_elec_new);
    var_v2 = var(v2_elec_new);
    v_elec_var_hist(steps+1) = ( 0.5*(var_v1 + var_v2) );

    Ex_L2_hist(steps+1) = get_L_2_error(E1,zeros(size(B3)),dx*dy);
    Ey_L2_hist(steps+1) = get_L_2_error(E2,zeros(size(B3)),dx*dy);
    Bz_L2_hist(steps+1) = get_L_2_error(B3,zeros(size(B3)),dx*dy);

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

    rho_hist(steps) = sum(sum(rho_mesh(1:end-1,1:end-1)));

end

temp_hist = ((M_electron * V^2) / k_B) * v_elec_var_hist;

ts = 0:dt:(N_steps-1)*dt;

gauge_error_array = zeros(length(ts),3);
gauge_error_array(:,1) = ts;
gauge_error_array(:,2) = gauge_error_L2;
gauge_error_array(:,3) = gauge_error_inf;

gauss_error_array = zeros(length(ts),3);
gauss_error_array(:,1) = ts;
gauss_error_array(:,2) = gauss_law_error_L2;
gauss_error_array(:,3) = gauss_law_error_inf;

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