function [gauge_error, gauss_law_error, sum_gauss_law_residual, v_elec_var_history, A1, A2, psi, rho_mesh, x1_elec_new, x2_elec_new] = ...
    asym_euler_particle_heating_solver(x1_ions_in, x2_ions_in, ...
                                       P1_ions_in, P2_ions_in, ...
                                       v1_ions_in, v2_ions_in, ...
                                       x1_elec_in, x2_elec_in, ...
                                       P1_elec_in, P2_elec_in, ...
                                       v1_elec_in, v2_elec_in, ...
                                       x, y, dx, dy, kappa, T_final, N_steps, ...
                                       q_ions, q_elec, ...
                                       r_ions, r_elec, ...
                                       w_ions, w_elec, ...
                                       sigma_1,sigma_2, ...
                                       results_path, ...
                                       enable_plots, ...
                                       plot_at)
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
    vidPath = projectRoot + "\results\conserving\" + tag + "\";
    disp(vidPath);
%     vidName = "potentials" + ".mp4";
    vidName = "moving_electron_bulk" + ".mp4";
    vidObj = VideoWriter(vidPath + vidName, 'MPEG-4');
    open(vidObj);

    figure;
    x0=200;
    y0=100;
    width = 1200;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height])
    
    
    % Make a list for tracking the electron velocity history
    % we use this to compute the temperature outside the solver
    % This variance is an average of the variance in each direction
    v_elec_var_history = [];

    % Grid dimensions
    N_x = length(x);
    N_y = length(y);
    
    % Domain lengths
    L_x = x(end) - x(1);
    L_y = y(end) - y(1);
    
    % Compute the step size
    dt = T_final/N_steps;
    
    % MOLT stability parameter
    % Set for the first-order method
    beta_BDF = 1.0;
    
    %------------------------------------------------------------------
    % Storage for the integrator
    %------------------------------------------------------------------

    % Initial position, momentum, and velocity of the particles
    % We copy the input data rather than overwrite it
    % and we store two time levels of history
    %
    % We'll assume that the ions remain stationary
    % so that we only need to update electrons.
    
    % Ion positions
    x1_ions = x1_ions_in(:);
    x2_ions = x2_ions_in(:);
    
    % Ion momenta
    P1_ions = P1_ions_in(:);
    P2_ions = P2_ions_in(:);
    
    % Ion velocities
    v1_ions = v1_ions_in(:);
    v2_ions = v2_ions_in(:);
    
    % Electron positions
    x1_elec_old = x1_elec_in(:);
    x2_elec_old = x2_elec_in(:);
    
    x1_elec_new = x1_elec_in(:);
    x2_elec_new = x2_elec_in(:);
    
    % Electron momenta
    P1_elec_old = P1_elec_in(:);
    P2_elec_old = P2_elec_in(:);
    
    P1_elec_new = P1_elec_in(:);
    P2_elec_new = P2_elec_in(:);
    
    % Electron velocities
    v1_elec_old = v1_elec_in(:);
    v2_elec_old = v2_elec_in(:);
    
    v1_elec_new = v1_elec_in(:);
    v2_elec_new = v2_elec_in(:);
    
    % Velocity at time t^{n-1} used for the Taylor approx. 
    v1_elec_nm1 = v1_elec_in(:);
    v2_elec_nm1 = v2_elec_in(:);
    
    % Taylor approximated velocity
    % v_star = v^{n} + ddt(v^{n})*dt
    % which is approximated by
    % v^{n} + (v^{n} - v^{n-1})
    v1_elec_star = v1_elec_in(:);
    v2_elec_star = v2_elec_in(:);
    
    % Store the total number of particles for each species
    N_ions = length(x1_ions);
    N_elec = length(x1_elec_new);
    
    % Mesh/field data
    % Need psi, A1, and A2
    % as well as their derivatives
    %
    % We compute ddt_psi with backwards differences
    psi = zeros(N_y,N_x,3);
    ddx_psi = zeros(N_y,N_x);
    ddy_psi = zeros(N_y,N_x);
    psi_src = zeros(N_y,N_x);
    
    A1 = zeros(N_y, N_x, 3);
    ddx_A1 = zeros(N_y,N_x);
    ddy_A1 = zeros(N_y,N_x);
    A1_src = zeros(N_y,N_x);
    
    A2 = zeros(N_y, N_x, 3);
    ddx_A2 = zeros(N_y,N_x);
    ddy_A2 = zeros(N_y,N_x);
    A2_src = zeros(N_y,N_x);
    
    % Other data needed for the evaluation of 
    % the gauge and Gauss' law
    ddt_psi = zeros(N_y,N_x);
    ddt_A1 = zeros(N_y,N_x);
    ddt_A2 = zeros(N_y,N_x);
    
    E1 = zeros(N_y,N_x);
    E2 = zeros(N_y,N_x);
    
    % Note that from the relation B = curl(A), we identify
    % B3 = ddx(A2) - ddy(A1)
    B3 = zeros(N_y,N_x);
    
    ddx_E1 = zeros(N_y,N_x);
    ddy_E2 = zeros(N_y,N_x);
    
    gauge_residual = zeros(N_y,N_x);
    gauss_law_residual = zeros(N_y,N_x);
    
    gauge_error = zeros(N_steps,1);
    gauss_law_error = zeros(N_steps,1);
    sum_gauss_law_residual = zeros(N_steps,1);

    % Storage for the particle data on the mesh
    rho_ions = zeros(N_y,N_x);
    rho_elec = zeros(N_y,N_x);
    rho_mesh = zeros(N_y,N_x);

    rho_elec_next = zeros(N_y,N_x);

%     u_avg_mesh = zeros(2,N_x,N_y);
%     u_avg_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_elec_new, x2_elec_new, v1_elec_old, v2_elec_old);
    
    % We track two time levels of J (n, n+1)
    % Note, we don't need J3 for this model 
    % Since ions are stationary J_mesh := J_elec
    J_mesh = zeros(N_y,N_x,2); % Idx order: grid indices (y,x), time level
    
    J1_mesh_FFTx = zeros(N_x-1,N_y-1);
    J2_mesh_FFTy = zeros(N_x-1,N_y-1);

    Jx_deriv_star = zeros(N_y,N_x);
    Jy_deriv_star = zeros(N_y,N_x);

    J1_clean = zeros(N_y,N_x);
    J2_clean = zeros(N_y,N_x);
    J1_deriv_clean = zeros(N_y,N_x);
    J2_deriv_clean = zeros(N_y,N_x);

    kx = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];
    ky = 2*pi/(L_y)*[0:(N_y-1)/2-1, 0, -(N_y-1)/2+1:-1];

    % ddx_J1 = zeros(N_y,N_x);
    % ddy_J2 = zeros(N_y,N_x);
    
    % Compute the cell volumes required in the particle to mesh mapping
    % The domain is periodic here, so the first and last cells here are
    % identical.
    cell_volumes = dx*dy*ones(N_y,N_x);
        
    % Current time of the simulation and step counter
    t_n = 0.0;
    steps = 1;

    csv_path = fullfile(results_path, "csv_files");
    figures_path = fullfile(results_path, "figures");

    rho_plot_path = fullfile(figures_path,"rho-plot");
    J_plot_path = fullfile(figures_path,"J-plot");
    A1_plot_path = fullfile(figures_path,"A1-plot");
    A2_plot_path = fullfile(figures_path,"A2-plot");
    psi_plot_path = fullfile(figures_path,"phi-plot");
    gauge_slice_plot_path = fullfile(figures_path,"gauge-plot","slice");
    gauge_surface_plot_path = fullfile(figures_path,"gauge-plot","surface");
    gauss_slice_plot_path = fullfile(figures_path,"gauss-plot","slice");
    gauss_surface_plot_path = fullfile(figures_path,"gauss-plot","surface");
    E_plot_path = fullfile(figures_path,"E-plot");
    B_plot_path = fullfile(figures_path,"B-plot");

    if enable_plots

        results_paths = [rho_plot_path, J_plot_path, A1_plot_path, A2_plot_path, ...
                        psi_plot_path, gauge_slice_plot_path, gauge_surface_plot_path, ...
                        gauss_slice_plot_path, gauss_surface_plot_path, E_plot_path, B_plot_path];
        for path = results_paths
            if ~exist(path)
                mkdir(path)
            end
        end
    end

    % Ions
    rho_ions = map_rho_to_mesh_2D(x, y, dx, dy, ...
                                  x1_ions, x2_ions, ...
                                  q_ions, cell_volumes, w_ions);

    % Electrons
    rho_elec = map_rho_to_mesh_2D(x, y, dx, dy, ...
                                  x1_elec_new, x2_elec_new, ...
                                  q_elec, cell_volumes, w_elec);
    % Need to enforce periodicity for the charge on the mesh
    rho_ions = enforce_periodicity(rho_ions(:,:));
    rho_elec = enforce_periodicity(rho_elec(:,:));
    
    rho_mesh = rho_ions + rho_elec;
    
    v_elec_var_history = zeros(N_steps, 1);

    rho_hist = zeros(N_steps,1);
    rho_hist(steps) = sum(sum(rho_elec(1:end-1,1:end-1)));

%     J_rho_update_type_vanilla = 'J_rho_update_vanilla';
%     J_rho_update_type_fft = 'J_rho_update_fft';
%     J_rho_update_fft_iterative = 'J_rho_update_fft_iterative'
%     eval(J_rho_update_type_fft);
    
    while(steps < N_steps)
        
        v1_elec_old = ramp(t_n,kappa)*v1_elec_old; % + ramp_velocity(t_n);
        v2_elec_old = ramp(t_n,kappa)*v2_elec_old; % + ramp_velocity(t_n);

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

%         J_rho_update_vanilla;
        J_rho_update_fft;
%         J_rho_update_fft_iterative;
        %---------------------------------------------------------------------
        % 5. Advance the psi and its derivatives by dt using BDF-1 
        %---------------------------------------------------------------------
        
        psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:);
        
        % Charge density is at the new time level from step (3)
        % which is consistent with the BDF scheme
        [psi, ddx_psi, ddy_psi] = BDF1_combined_per_advance(psi, ddx_psi, ddy_psi, psi_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF);
        assert(all(abs(psi(1,:,3) - psi(end,:,3)) < 10*eps));
        assert(all(abs(psi(:,1,3) - psi(:,end,3)) < 10*eps));
        assert(all(abs(ddx_psi(1,:) - ddx_psi(end,:)) < 10*eps));
        assert(all(abs(ddx_psi(:,1) - ddx_psi(:,end)) < 10*eps));
        assert(all(abs(ddy_psi(1,:) - ddy_psi(end,:)) < 10*eps));
        assert(all(abs(ddy_psi(:,1) - ddy_psi(:,end)) < 10*eps));
        % Wait to shuffle until the end, but we could do that here
        
        %---------------------------------------------------------------------
        % 5. Advance the A1 and A2 and their derivatives by dt using BDF-1
        %---------------------------------------------------------------------
        
        A1_src(:,:) = sigma_2*J1_mesh;
        A2_src(:,:) = sigma_2*J2_mesh;
        % A1_src(:,:) = sigma_2*J1_star;
        % A2_src(:,:) = sigma_2*J2_star;
        % A1 uses J1
        [A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                         x, y, t_n, dx, dy, dt, kappa, beta_BDF);
        
        % A2 uses J2
        [A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                         x, y, t_n, dx, dy, dt, kappa, beta_BDF);
        
%         clean_splitting_error;
                                                     
        assert(all(abs(A1(1,:,3) - A1(end,:,3)) < 10*eps));
        assert(all(abs(A1(:,1,3) - A1(:,end,3)) < 10*eps));
        assert(all(abs(ddx_A1(1,:) - ddx_A1(end,:)) < 10*eps));
        assert(all(abs(ddx_A1(:,1) - ddx_A1(:,end)) < 10*eps));
        assert(all(abs(ddy_A1(1,:) - ddy_A1(end,:)) < 10*eps));
        assert(all(abs(ddy_A1(:,1) - ddy_A1(:,end)) < 10*eps));
        
        assert(all(abs(A2(1,:,3) - A2(end,:,3)) < 10*eps));
        assert(all(abs(A2(:,1,3) - A2(:,end,3)) < 10*eps));
        assert(all(abs(ddx_A2(1,:) - ddx_A2(end,:)) < 10*eps));
        assert(all(abs(ddx_A2(:,1) - ddx_A2(:,end)) < 10*eps));
        assert(all(abs(ddy_A2(1,:) - ddy_A2(end,:)) < 10*eps));
        assert(all(abs(ddy_A2(:,1) - ddy_A2(:,end)) < 10*eps));
        % Wait to shuffle until the end, but we could do that here
        
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
        
        gauge_error(steps) = get_L_2_error(gauge_residual(:,:), ...
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
        
        gauss_law_error(steps) = get_L_2_error(gauss_law_residual(:,:), ...
                                               zeros(size(gauss_law_residual(:,:))), ...
                                               dx*dy);
        
        % Now we measure the sum of the residual in Gauss' law (avoiding the boundary)
        sum_gauss_law_residual(steps) = sum(sum(gauss_law_residual(:,:)));
        
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
        v_elec_var_history(steps) = ( 0.5*(var_v1 + var_v2) );
        
        rho_hist(steps) = sum(sum(rho_elec(1:end-1,1:end-1)));

        if (mod(steps-1, plot_at) == 0)
            subplot(2,2,1);
            scatter(x1_elec_new, x2_elec_new, 5, 'filled');
            xlabel("x");
            ylabel("y");
            title("Electron Locations");
            xlim([x(1),x(end)]);
            ylim([y(1),y(end)]);

            subplot(2,2,2);
            surf(x,y,rho_mesh);
            xlabel("x");
            ylabel("y");
            title("$\rho$",'Interpreter','latex');
            xlim([x(1),x(end)]);
            ylim([y(1),y(end)]);

            % subplot(2,2,3);
            % surf(x,y,psi(:,:,3));
            % xlabel("x");
            % ylabel("y");
            % title("$\psi$",'Interpreter','latex');
            % xlim([x(1),x(end)]);
            % ylim([y(1),y(end)]);

%             subplot(2,2,3);
%             plot(0:dt:(steps-1)*dt,rho_hist(1:steps));
%             xlabel("t");
%             ylabel("$\sum_{i,j}\rho_{i,j}$",'Interpreter','latex');
%             title("Total Charge Over Time");

            subplot(2,2,3);
            surf(x,y,A2(:,:,3));
            xlabel("x");
            ylabel("y");
            title("$A_2$",'Interpreter','latex');
            xlim([x(1),x(end)]);
            ylim([y(1),y(end)]);

            subplot(2,2,4);
            surf(x,y,gauge_residual);
            xlabel("x");
            ylabel("y");
            title("Gauge Error");
            xlim([x(1),x(end)]);
            ylim([y(1),y(end)]);

            sgtitle("FFT method, t = " + t_n);
            
            drawnow;

            currFrame = getframe(gcf);
            writeVideo(vidObj, currFrame);
        end

        % Step is now complete
        steps = steps + 1;
        t_n = t_n + dt;

    end

    close(vidObj);
    figure;
    plot(0:dt:(N_steps-2)*dt, gauge_error(1:end-1));
    xlabel("t");
    ylabel("Gauge Error");
    title("FFT Gauge Error over Time");
end

function r = ramp(t,kappa)
%     r = kappa/100*exp(-((time - .05)^2)/.00025);
    r = 1;
    if t < .1
        r = sin((2*pi*t)/.01);
    end
end