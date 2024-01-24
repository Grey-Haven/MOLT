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


    
    %---------------------------------------------------------------------
    % 5. Correct gauge error (optional)
    %---------------------------------------------------------------------
%     clean_splitting_error;
%     gauge_correction_FFT_deriv;
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
    ddx_E1 = compute_ddx_FD(E1, dx);
    ddy_E2 = compute_ddy_FD(E2, dy);
    
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
    
    LHS = (1/(kappa^2))*ddt2_phi - laplacian_phi_FFT;
    RHS = rho_mesh / sigma_1;

    alpha = kappa*dt;

    LHS_alt0 = 1/(alpha^2) * (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2)) - laplacian_phi_FFT;
    res0 = LHS - RHS;

    LHS_alt = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2)) - alpha^2 * laplacian_phi_FFT;
    RHS_alt = alpha^2 * rho_mesh / sigma_1;
    res_alt = LHS_alt - RHS_alt;

    res = LHS - RHS;
    divE = - laplacian_phi_FFT - ddt_div_A;
    gauss_law_alt_residual = divE - rho_mesh;
    gauss_law_alt_err = get_L_2_error(gauss_law_alt_residual, ...
                                      zeros(size(gauss_law_residual(:,:))), ...
                                      dx*dy);

    subplot(1,2,1);
    surf(x,y,res);
    title("$\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta t^2} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);
    subplot(1,2,2);
    surf(x,y,res_alt);
    title("$\phi^{n+1} - 2\phi^{n} + \phi^{n-1} - \alpha^2\Delta \phi^{n+1} - \alpha^2\frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);

    
    subplot(1,2,1);
    surf(x,y,-ddt_div_A - laplacian_phi_FFT - rho_mesh / sigma_1);
    title("$-\frac{\nabla \cdot \textbf{A}^{n+1} - \nabla \cdot \textbf{A}^{n}}{\partial t} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);
    subplot(1,2,2);
    surf(x,y,-alpha^2*ddt_div_A - alpha^2*laplacian_phi_FFT - alpha^2*rho_mesh / sigma_1);
    title("$-\alpha^2\frac{\nabla \cdot \textbf{A}^{n+1} - \nabla \cdot \textbf{A}^{n}}{\partial t} - \alpha^2\Delta \phi^{n+1} - \alpha^2\frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);

    subplot(2,2,1);
    surf(x,y,LHS);
    title("$\frac{1}{\kappa^2}\frac{\partial^2 \phi}{\partial t^2} - \Delta \phi$",'interpreter','latex','FontSize',24);
    view(2);
    colorbar;
    axis square;
    subplot(2,2,2);
    surf(x,y,RHS);
    title("$\frac{\rho}{\sigma_1}$",'interpreter','latex','FontSize',24);
    view(2);
    colorbar;
    axis square;
    subplot(2,2,3);
    surf(x,y,res);
    title("$\left(\frac{1}{\kappa^2}\frac{\partial^2 \phi}{\partial t^2} - \Delta \phi\right) - \frac{\rho}{\sigma_1}$",'interpreter','latex','FontSize',24);
    view(2);
    colorbar;
    axis square;
    subplot(2,2,4);
    scatter(x1_elec_new, x2_elec_new, 5, 'filled');
    xlabel("x");
    ylabel("y");
    title("Electron Locations");
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;

    sgtitle({update_method_title + " method", "Grid: " + tag + ", CFL: " + CFL + ", Particle Multiplier: " + particle_count_multiplier, "t = " + num2str(t_n,'%.4f')});
    drawnow;

    
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

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots;
    close(vidObj);
end
writematrix(gauge_error_array,csvPath + "gauge_error.csv");

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