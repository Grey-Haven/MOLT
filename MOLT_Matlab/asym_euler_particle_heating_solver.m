%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle solver for the 2D-2P heating test that uses the asymmetrical Euler method for particles
% and the MOLT field solvers.
%
% Note that this problem starts out as charge neutral and with a net zero current. Therefore, the
% fields are taken to be zero initially.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tag = (length(x)-1) + "x" + (length(y)-1);
filePath = matlab.desktop.editor.getActiveFilename;
projectRoot = fileparts(filePath);

% modification = "no_mod";
modification = "correct_gauge";

resultsPath = projectRoot + "/results/conserving/p_mult_" + particle_count_multiplier + ...
              "/" + tag + "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/";
figPath = resultsPath + "figures/";
csvPath = resultsPath + "csv_files/";
disp(resultsPath);
create_directories;

if enable_plots
    vidName = "moving_electron_bulk" + ".mp4";
    vidObj = VideoWriter(figPath + vidName, 'MPEG-4');
    open(vidObj);
    
    figure;
    x0=200;
    y0=100;
    width = 1200;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height])
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
    
    v1_elec_old = v1_elec_old + ramp_drift(t_n)*v1_drift;
    v2_elec_old = v2_elec_old + ramp_drift(t_n)*v2_drift;

    %---------------------------------------------------------------------
    % 1. Advance electron positions by dt using v^{n}
    %---------------------------------------------------------------------
     
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_new, x2_elec_new, ...
                                                               x1_elec_old, x2_elec_old, ...
                                                               v1_elec_old, v2_elec_old, dt);
    
    % Apply the particle boundary conditions
    % Need to include the shift function here
    x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
    x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);
    %---------------------------------------------------------------------

    %---------------------------------------------------------------------
    % 2. Compute the electron current density used for updating A
    %    Compute also the charge density used for updating psi
    %---------------------------------------------------------------------

    J_rho_update_vanilla;
%     J_rho_update_fft;
%     J_rho_update_fft_iterative;

    %---------------------------------------------------------------------
    % 5. Advance the psi and its derivatives by dt using BDF-1 
    %---------------------------------------------------------------------
    
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:);
    
    % Charge density is at the new time level from step (3)
    % which is consistent with the BDF scheme
    [psi, ddx_psi, ddy_psi] = BDF1_combined_per_advance(psi, ddx_psi, ddy_psi, psi_src(:,:), ...
                                                        x, y, t_n, dx, dy, dt, kappa, beta_BDF);

    %---------------------------------------------------------------------
    % 5. Advance the A1 and A2 and their derivatives by dt using BDF-1
    %---------------------------------------------------------------------

    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    % A1 uses J1
    [A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                     x, y, t_n, dx, dy, dt, kappa, beta_BDF);
    
    % A2 uses J2
    [A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                     x, y, t_n, dx, dy, dt, kappa, beta_BDF);


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
                                           x, y, dx, dy, q_elec, r_elec, dt);
                                       
    %---------------------------------------------------------------------
    % 7. Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) )/dt;
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:) + ddx_A1(:,:) + ddy_A2(:,:);
    
    gauge_error(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                         zeros(size(gauge_residual(:,:))), ...
                                         dx*dy);
    
    % Compute the ddt_A with backwards finite-differences
    ddt_A1(:,:) = ( A1(:,:,end) - A1(:,:,end-1) )/dt;
    ddt_A2(:,:) = ( A2(:,:,end) - A2(:,:,end-1) )/dt;
    
    % Compute E = -grad(psi) - ddt_A
    % For ddt A, we use backward finite-differences
    % Note, E3 is not used in the particle update so we don't need ddt_A3
    E1(:,:) = -ddx_psi(:,:) - ddt_A1(:,:);
    E2(:,:) = -ddy_psi(:,:) - ddt_A2(:,:);
    
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
    % 7.5 Correct gauge error
    %---------------------------------------------------------------------
    %     clean_splitting_error;
    gauge_correction;

    
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

    % u_prev(:,:) = u_star(:,:);
    
    % % Measure the variance of the electron velocity distribution
    % and store for later use
    %
    % Note that we average the variance here so we don't need an
    % extra factor of two outside of this function
    var_v1 = var(v1_elec_new);
    var_v2 = var(v2_elec_new);
    v_elec_var_history(steps+1) = ( 0.5*(var_v1 + var_v2) );

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
gauge_error_array = zeros(length(ts),2);
gauge_error_array(:,1) = ts;
gauge_error_array(:,2) = gauge_error;

if (write_csvs)
    save_csvs;
end
if enable_plots
    create_plots;
    close(vidObj);
end
writematrix(gauge_error_array,csvPath+"gauge_error.csv");

% figure;
% plot(0:dt:(N_steps-1)*dt, gauge_error);
% xlabel("t");
% ylabel("Gauge Error");
% title({'Gauge Error Over Time', update_method_title,tag + ", CFL: " + CFL});

% filename = figPath + "gauge_error.jpg";

% saveas(gcf,filename)

function r = ramp(t,kappa)
%     r = kappa/100*exp(-((time - .05)^2)/.00025);
    r = 1;
    if t < .1
        r = sin((2*pi*t)/.01);
    end
end

function r = ramp_drift(t)
    r = exp(-500*(t-.1).^2);
end