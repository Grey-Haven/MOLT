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
    [x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
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

    if J_rho_update_method == J_rho_update_method_vanilla
        J_rho_update_vanilla;
    elseif ismember(J_rho_update_method, J_rho_BDF_FFT_Family)
        J_rho_update_fft;
    elseif J_rho_update_method == J_rho_update_method_FD6
        J_rho_update_FD6;
    else
        ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
        throw(ME);
    end
    
    %---------------------------------------------------------------------
    % 3.1. Compute wave sources
    %---------------------------------------------------------------------
    psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:,end);
    A1_src(:,:) = sigma_2*J1_mesh;
    A2_src(:,:) = sigma_2*J2_mesh;

    %---------------------------------------------------------------------
    % 3.2 Update the scalar (phi) and vector (A) potentials waves. 
    %---------------------------------------------------------------------
    if waves_update_method == waves_update_method_vanilla
        update_waves;
    elseif waves_update_method == waves_update_method_FFT
        update_waves_hybrid_FFT;
        % update_waves_pure_FFT;
    elseif waves_update_method == waves_update_method_FD6
        update_waves_hybrid_FD6;
    elseif waves_update_method == waves_update_method_poisson_phi
        update_waves_poisson_phi;
    elseif waves_update_method == waves_update_method_pure_FFT
        update_waves_pure_FFT;
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
    % 5. Compute the errors in the Lorenz gauge and Gauss' law
    %---------------------------------------------------------------------
    
    % Compute the time derivative of psi using finite differences
    ddt_psi(:,:) = ( psi(:,:,end) - psi(:,:,end-1) ) / dt;
    
    % Compute the residual in the Lorenz gauge 
    gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);
    
    gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                            zeros(size(gauge_residual(:,:))), ...
                                            dx*dy);
    gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));

    rho_hist(steps+1) = sum(sum(rho_elec(1:end-1,1:end-1)));

    % compute_gauss_residual;

    
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

    rho_mesh = shuffle_steps(rho_mesh);

    % Step is now complete
    steps = steps + 1;
    t_n = t_n + dt;

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

writematrix(gauge_error_array,csvPath + "gauge_error.csv");
writematrix(gauss_error_array,csvPath + "gauss_error.csv");

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

potential_label = "$\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta t^2} - \Delta \left(\phi^{n+1} + \phi^{n-1}\right) - \frac{1}{\sigma_1}\frac{\rho^{n+1} + \rho^{n-1}}{2}$";
gauge_label = "$-\frac{\nabla\cdot\textbf{A}^{n+1} - \nabla\cdot\textbf{A}^{n}}{\Delta t} - \Delta \left(\phi^{n+1} + \phi^{n-1}\right) - \frac{1}{\sigma_1}\frac{\rho^{n+1} + \rho^{n-1}}{2}$";
field_label = "$\nabla\cdot\textbf{E}^{n+1} - \frac{1}{\sigma_1}\frac{\rho^{n+1} + \rho^{n-1}}{2}$";
legend([potential_label,gauge_label,field_label],'FontSize',24,'interpreter','latex','location','east');
title("Gauss' Law",'FontSize',32);

subplot(1,2,2);
plot(ts,gauge_error_L2)
title("Gauge Error",'FontSize',32);

sgtitle("Hybrid MOLT-BDF1 Wave, FFT Derivatives, " + tag,'FontSize',48);

saveas(gcf,figPath + "residuals.jpg");

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