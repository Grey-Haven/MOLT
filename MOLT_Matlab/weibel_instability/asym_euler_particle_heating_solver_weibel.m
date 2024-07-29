%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle solver for the 2D-2P heating test that uses the asymmetrical Euler method for particles
% and the MOLT field solvers.
%
% Note that this problem starts out as charge neutral and with a net zero current. Therefore, the
% fields are taken to be zero initially.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while(steps < N_steps)
    
%     v1_elec_old = ramp(t_n)*v1_elec_old; % + ramp_drift(t_n)*v1_drift;
%     v2_elec_old = ramp(t_n)*v2_elec_old; % + ramp_drift(t_n)*v2_drift;

    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------

    v1_star = 2*v1_elec_old - v1_elec_nm1;
    v2_star = 2*v2_elec_old - v2_elec_nm1;
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
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

    % J_rho_update_vanilla;
%     J_rho_update_fft;
    J_rho_update_FD6;
    % J_rho_update_fft_iterative;    
    
    %---------------------------------------------------------------------
    % 5.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:,end);
    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    %---------------------------------------------------------------------
    % 5.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
%     update_waves;
%     update_waves_hybrid_BDF;
%     update_waves_hybrid_FFT;
    update_waves_hybrid_FD6;
    % update_waves_FFT_alt;
%     update_waves_FFT;


    
    %---------------------------------------------------------------------
    % 5.5 Correct gauge error
    %---------------------------------------------------------------------
%     clean_splitting_error;
%     gauge_correction_FFT_deriv;
    % gauge_correction_FD6_deriv;
    

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
                                           ddx_psi, ddy_psi, ...
                                           A1(:,:,end), ddx_A1, ddy_A1, ...
                                           A2(:,:,end), ddx_A2, ddy_A2, ...
                                           x, y, dx, dy, q_elec, r_elec, ...
                                           kappa, dt);


    %---------------------------------------------------------------------
    % 7. Diagnostics and Storage
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 7.1 Compute the E and B fields
    %---------------------------------------------------------------------

    % Compute E = -grad(psi) - ddt_A
    % For ddt A, we use backward finite-differences
    % Note, E3 is not used in the particle update so we don't need ddt_A3
    E1(:,:) = -ddx_psi(:,:) - ddt_A1(:,:);
    E2(:,:) = -ddy_psi(:,:) - ddt_A2(:,:);
        
    % Compute B = curl(A)
    B3 = ddx_A2(:,:,end) - ddy_A1(:,:,end);

    %---------------------------------------------------------------------
    % 7.2 Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) )/dt;
    
    % Compute the ddt_A with backwards finite-differences
    ddt_A1(:,:) = ( A1(:,:,end) - A1(:,:,end-1) )/dt;
    ddt_A2(:,:) = ( A2(:,:,end) - A2(:,:,end-1) )/dt;
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:) + ddx_A1(:,:) + ddy_A2(:,:);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));

    %-----------------------------------------------------------------------
    % 7.3 Compute and store diagnostics
    %-----------------------------------------------------------------------
    total_mass_ions = compute_total_mass_species(rho_ions(1:end-1,1:end-1), cell_volumes(1:end-1,1:end-1), q_ions, r_ions);
    total_mass_elec = compute_total_mass_species(rho_elec(1:end-1,1:end-1), cell_volumes(1:end-1,1:end-1), q_elec, r_elec);
    
    [kinetic_energy_ions, ...
     potential_energy_ions, ...
     total_energy_ions] = compute_total_energy(psi, A1, A2, ...
                                             x1_ions, x2_ions, ...
                                             P1_ions, P2_ions, ...
                                             x, y, q_ions, r_ions);
    
    [kinetic_energy_elec, ...
     potential_energy_elec, ...
     total_energy_elec] = compute_total_energy(psi, A1, A2, ...
                                             x1_elec_new, x2_elec_new, ...
                                             P1_elec_new, P2_elec_new, ...
                                             x, y, q_elec, r_elec);
    
    % Combine the results from both species
    total_kinetic(steps+1) = kinetic_energy_ions + kinetic_energy_elec;
    total_potential(steps+1) = potential_energy_ions + potential_energy_elec;
    total_energy(steps+1) = total_energy_ions + total_energy_elec;

    total_mass(steps+1) = total_mass_ions + total_mass_elec;

    Ex_L2_hist(steps+1) = get_L_2_error(E1,zeros(size(B3)),dx*dy);
    Ey_L2_hist(steps+1) = get_L_2_error(E2,zeros(size(B3)),dx*dy);
    Bz_L2_hist(steps+1) = get_L_2_error(B3,zeros(size(B3)),dx*dy);

    rho_hist(steps+1) = sum(sum(rho_elec(1:end-1,1:end-1)));

    var_v1 = var(v1_elec_new);
    var_v2 = var(v2_elec_new);
    v_elec_var_history(steps+1) = ( 0.5*(var_v1 + var_v2) );

    p1_sum = sum(P1_elec_new);
    p2_sum = sum(P2_elec_new);
    p_norm = norm([p1_sum, p2_sum]);

    p_elec_hist(steps+1) = p_norm;
    
    %---------------------------------------------------------------------
    % 7.4 Compute the error in Gauss' Law
    %---------------------------------------------------------------------

    % Compute Gauss' law div(E) - rho to check the involution
    % We'll just use finite-differences here
    ddx_E1 = compute_ddx_FD(E1, dx);
    ddy_E2 = compute_ddy_FD(E2, dy);
    
    gauss_law_residual(:,:) = ddx_E1(:,:) + ddy_E2(:,:) - psi_src(:,:);
    
    gauss_law_error(steps+1) = get_L_2_error(gauss_law_residual(:,:), ...
                                           zeros(size(gauss_law_residual(:,:))), ...
                                           dx*dy);
    
    % Now we measure the sum of the residual in Gauss' law (avoiding the boundary)
    sum_gauss_law_residual(steps+1) = sum(sum(gauss_law_residual(:,:)));

    
    %---------------------------------------------------------------------
    % 8. Prepare for the next time step by shuffling the time history data
    %---------------------------------------------------------------------
    
    % Shuffle the time history of the fields
    psi = shuffle_steps(psi);
    A1 = shuffle_steps(A1);
    A2 = shuffle_steps(A2);
    
    % Shuffle the time history of the particle data
    v1_elec_nm1(:) = v1_elec_old(:);
    v2_elec_nm1(:) = v2_elec_old(:);
    
    x1_elec_old(:) = x1_elec_new(:);
    x2_elec_old(:) = x2_elec_new(:);
    
    v1_elec_old(:) = v1_elec_new(:);
    v2_elec_old(:) = v2_elec_new(:);
    
    P1_elec_old(:) = P1_elec_new(:);
    P2_elec_old(:) = P2_elec_new(:);


    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots_weibel;
            % create_plots_magnetic_magnitude;
        end
    end

end

ts = 0:dt:(N_steps-1)*dt;
gauge_error_array = zeros(length(ts),3);
gauge_error_array(:,1) = ts;
gauge_error_array(:,2) = gauge_error_L2;
gauge_error_array(:,3) = gauge_error_inf;

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

mass_hist_array = zeros(length(ts),2);
mass_hist_array(:,1) = ts;
mass_hist_array(:,2) = total_mass;

energy_hist_array = zeros(length(ts),4);
energy_hist_array(:,1) = ts;
energy_hist_array(:,2) = total_kinetic;
energy_hist_array(:,3) = total_potential;
energy_hist_array(:,4) = total_energy;

p_hist_array = zeros(length(ts),2);
p_hist_array(:,1) = ts;
p_hist_array(:,2) = p_elec_hist;

v_var_hist_array = zeros(length(ts),2);
v_var_hist_array(:,1) = ts;
v_var_hist_array(:,2) = v_elec_var_history;

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots_weibel;
    close(vidObj);
end

figure;
x0=200 + width;
y0=100;
set(gcf,'position',[x0,y0,width,height]);

semilogy(ts, abs((total_mass - total_mass(1)) ./ total_mass(1)));
xlabel("Angular Plasma Periods",'FontSize',24);
ylabel("$||\left(M(t) - M(0)\right)/M(0)||_2$",'interpreter','latex', 'FontSize', 24);
title("Relative Total Mass Over Time",'FontSize',32);
subtitle(update_method_title + " " + tag,'FontSize',24);

saveas(gcf,figPath + "total_mass" + tag + ".jpg");
close;

figure;
x0=200 + width;
y0=100;
set(gcf,'position',[x0,y0,width,height]);

semilogy(ts, abs((total_energy - total_energy(1)) ./ total_energy(1)));
xlabel("Angular Plasma Periods",'FontSize',24);
ylabel("$||\left(E(t) - E(0)\right)/E(0)||_2$",'interpreter','latex', 'FontSize', 24);
title("Relative Total Energy Over Time",'FontSize',32);
subtitle(update_method_title + " " + tag,'FontSize',24);

saveas(gcf,figPath + "total_energy" + tag + ".jpg");
close;

writematrix(gauge_error_array,csvPath + "gauge_error.csv");
writematrix(B3_L2_array,csvPath + "B3_magnitude.csv");
writematrix(E1_L2_array,csvPath + "E1_magnitude.csv");
writematrix(E2_L2_array,csvPath + "E2_magnitude.csv");
writematrix(rho_hist_array,csvPath + "rho_hist.csv");
writematrix(mass_hist_array,csvPath + "mass_hist.csv");
writematrix(energy_hist_array,csvPath + "energy_hist.csv");
writematrix(p_hist_array,csvPath + "p_hist.csv");
writematrix(v_var_hist_array,csvPath + "v_var_hist.csv");