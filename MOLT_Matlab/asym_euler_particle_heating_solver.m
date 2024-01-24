%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle solver for the 2D-2P heating test that uses the asymmetrical Euler method for particles
% and the MOLT field solvers.
%
% Note that this problem starts out as charge neutral and with a net zero current. Therefore, the
% fields are taken to be zero initially.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if enable_plots
    vidName = "moving_electron_bulk" + ".mp4";
    vidObj = VideoWriter(figPath + vidName, 'MPEG-4');
    open(vidObj);
    
    figure;
    x0=200;
    y0=100;
    width = 1800;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height]);
end

steps = 0;
if (write_csvs)
    save_csvs;
end
if (enable_plots)
    create_plots;
end

rho_hist(steps+1) = sum(sum(rho_elec(1:end-1,1:end-1)));

gauss_law_error = zeros(N_steps,1);
sum_gauss_law_residual = zeros(N_steps,1);

while(steps < N_steps)

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

    %---------------------------------------------------------------------
    % 2. Compute the electron current density used for updating A
    %    Compute also the charge density used for updating psi
    %---------------------------------------------------------------------

%     J_rho_update_vanilla;
    J_rho_update_fft;
    % J_rho_update_FD6;
    
    %---------------------------------------------------------------------
    % 3. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:);
    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    %---------------------------------------------------------------------
    % 4. Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
%     update_waves;
    update_waves_hybrid_FFT;
    % update_waves_hybrid_FD6;


    % Alternative way of solving phi

    div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
    div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
    ddt_div_A = (div_A_curr - div_A_prev)/dt;

    RHS = -(rho_mesh / sigma_1 + ddt_div_A);
    LHS = solve_poisson_FFT(RHS,kx_deriv_2,ky_deriv_2);
    psi(:,:,end) = LHS;

    psi_next_fft_x = fft(psi(1:end-1,1:end-1,end),N_x-1,2);
    psi_next_fft_y = fft(psi(1:end-1,1:end-1,end),N_y-1,1);

    ddx_psi_fft = zeros(N_y,N_x);
    ddy_psi_fft = zeros(N_y,N_x);

    ddx_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*psi_next_fft_x,N_x-1,2);
    ddy_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*psi_next_fft_y,N_y-1,1);

    ddx_psi_fft = copy_periodic_boundaries(ddx_psi_fft);
    ddy_psi_fft = copy_periodic_boundaries(ddy_psi_fft);

    ddx_psi(:,:,end) = ddx_psi_fft;
    ddy_psi(:,:,end) = ddy_psi_fft;

    % End Alternative Way


    
    %---------------------------------------------------------------------
    % 5. Correct gauge error (optional)
    %---------------------------------------------------------------------
%     clean_splitting_error;
    % gauge_correction_FFT_deriv;
%     gauge_correction_FD6_deriv;
    

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
    % 7. Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) ) / dt;
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf = max(max(abs(gauge_residual)));
    
    % Compute the ddt_A with backwards finite-differences
    ddt_A1(:,:) = ( A1(:,:,end) - A1(:,:,end-1) )/dt;
    ddt_A2(:,:) = ( A2(:,:,end) - A2(:,:,end-1) )/dt;
    
    % Compute E = -grad(psi) - ddt_A
    % For ddt A, we use backward finite-differences
    % Note, E3 is not used in the particle update so we don't need ddt_A3
    E1(:,:) = -ddx_psi(:,:,end) - ddt_A1(:,:);
    E2(:,:) = -ddy_psi(:,:,end) - ddt_A2(:,:);
    
    % Compute Gauss' law div(E) - rho to check the involution
    % We'll just use finite-differences here
    % ddx_E1 = compute_ddx_FD(E1, dx);
    % ddy_E2 = compute_ddy_FD(E2, dy);
    ddx_E1 = compute_ddx_FFT(E1, kx_deriv_1);
    ddy_E2 = compute_ddy_FFT(E2, ky_deriv_1);

    LHS_field = ddx_E1(:,:) + ddy_E2(:,:);
    
    gauss_law_residual(:,:) = ddx_E1(:,:) + ddy_E2(:,:) - psi_src(:,:);
    
    gauss_law_error(steps+1) = get_L_2_error(gauss_law_residual(:,:), ...
                                           zeros(size(gauss_law_residual(:,:))), ...
                                           dx*dy);
    
    % Now we measure the sum of the residual in Gauss' law (avoiding the boundary)
    sum_gauss_law_residual(steps+1) = sum(sum(gauss_law_residual(:,:)));

    div_A_curr = ddx_A1(:,:,end) + ddy_A2(:,:,end);
    div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
    ddt_div_A = (div_A_curr - div_A_prev)/dt;

    ddt2_phi = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2))/(dt^2);

    laplacian_phi_FFT = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);
    
    LHS_potential = (1/(kappa^2))*ddt2_phi - laplacian_phi_FFT;
    RHS = rho_mesh / sigma_1;

    LHS_gauge = -ddt_div_A - laplacian_phi_FFT;

    gauss_law_potential_res = LHS_potential  - RHS;
    gauss_law_gauge_res     = LHS_gauge      - RHS;
    gauss_law_field_res     = LHS_field      - RHS;

    gauss_law_potential_err_L2(steps+1) = get_L_2_error(gauss_law_potential_res, ...
                                                        zeros(size(gauss_law_residual(:,:))), ...
                                                        dx*dy);
    gauss_law_gauge_err_L2(steps+1) = get_L_2_error(gauss_law_gauge_res, ...
                                                    zeros(size(gauss_law_residual(:,:))), ...
                                                    dx*dy);
    gauss_law_field_err_L2(steps+1) = get_L_2_error(gauss_law_field_res, ...
                                                    zeros(size(gauss_law_residual(:,:))), ...
                                                    dx*dy);

    gauss_law_potential_err_inf(steps+1) = max(max(abs(gauss_law_potential_res)));
    gauss_law_gauge_err_inf(steps+1) = max(max(abs(gauss_law_gauge_res)));
    gauss_law_field_err_inf(steps+1) = max(max(abs(gauss_law_field_res)));

    
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

    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

    rho_hist(steps) = sum(sum(rho_elec(1:end-1,1:end-1)));

    if (mod(steps, plot_at) == 0)
        if (write_csvs)
            save_csvs;
        end
        if (enable_plots)
            create_plots;
        end
    end

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
    create_plots;
    close(vidObj);
end

figure;
plot(ts, gauss_law_potential_err_L2);
xlabel("t");
title("$||\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta^2} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}||_2$", 'FontSize', 24);

figure;
plot(ts, gauss_law_gauge_err_L2);
xlabel("t");
title("$||-\frac{\nabla \cdot \textbf{A}^{n+1} - \nabla \cdot \textbf{A}^{n}}{\Delta t} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}||_2$", 'FontSize', 24);

figure;
plot(ts, gauss_law_field_err_L2);
xlabel("t");
title("$||\nabla \cdot \textbf{E} - \frac{\rho}{\sigma_1}||_2$", 'FontSize', 24);

writematrix(gauge_error_array,csvPath + "gauge_error.csv");
writematrix(gauss_error_array,csvPath + "gauss_error.csv");

% figure;
% plot(0:dt:(N_steps-1)*dt, gauge_error);
% xlabel("t");
% ylabel("Gauge Error");
% title({'Gauge Error Over Time', update_method_title,tag + ", CFL: " + CFL});

% filename = figPath + "gauge_error.jpg";

% saveas(gcf,filename)

function r = ramp(t)
%     r = kappa/100*exp(-((time - .05)^2)/.00025);
%     r = 1;
%     if t < .1
%         r = sin((2*pi*t)/.01);
%     end
    r = 1 - exp(-100 * t);
end

% function r = ramp_drift(t)
%     r = exp(-500*(t-.1).^2);
% end