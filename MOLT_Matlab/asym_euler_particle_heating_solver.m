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
resultsPath = projectRoot + "\results";
vidPath = resultsPath + "\conserving\" + tag + "\";
csvPath = resultsPath + "\conserving\" + tag + "\" + "csv_files\";
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

% Electron positions
x1_elec_old = x1_elec(:);
x2_elec_old = x2_elec(:);

x1_elec_new = x1_elec(:);
x2_elec_new = x2_elec(:);

% Electron momenta
P1_elec_old = P1_elec(:);
P2_elec_old = P2_elec(:);

P1_elec_new = P1_elec(:);
P2_elec_new = P2_elec(:);

% Electron velocities
v1_elec_old = v1_elec(:);
v2_elec_old = v2_elec(:);

v1_elec_new = v1_elec(:);
v2_elec_new = v2_elec(:);

% Velocity at time t^{n-1} used for the Taylor approx. 
v1_elec_nm1 = v1_elec(:);
v2_elec_nm1 = v2_elec(:);

% Taylor approximated velocity
% v_star = v^{n} + ddt(v^{n})*dt
% which is approximated by
% v^{n} + (v^{n} - v^{n-1})
v1_elec_star = v1_elec(:);
v2_elec_star = v2_elec(:);

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

% Current
J_mesh = map_J_to_mesh_2D2V(J_mesh(:,:,:), x, y, dx, dy, ...
                        x1_elec_new, x2_elec_new, ...
                        v1_elec_old, v2_elec_old, ...
                        q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

J1_mesh = J_mesh(:,:,1);
J2_mesh = J_mesh(:,:,2);

v_elec_var_history = zeros(N_steps, 1);

rho_hist = zeros(N_steps,1);

steps = 0;
if (write_csvs)
    save_csvs;
end
create_plots;

rho_hist(steps+1) = sum(sum(rho_elec(1:end-1,1:end-1)));

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

    J_rho_update_vanilla;
%         J_rho_update_fft;
%         J_rho_update_fft_iterative;

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
    % A1_src(:,:) = sigma_2*J1_star;
    % A2_src(:,:) = sigma_2*J2_star;
    % A1 uses J1
    [A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                     x, y, t_n, dx, dy, dt, kappa, beta_BDF);
    
    % A2 uses J2
    [A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                     x, y, t_n, dx, dy, dt, kappa, beta_BDF);
    
%         clean_splitting_error;
    
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
        create_plots;
    end

end

close(vidObj);
figure;
plot(0:dt:(N_steps-1)*dt, gauge_error);
xlabel("t");
ylabel("Gauge Error");
title("FFT Iterative Gauge Error over Time");

function r = ramp(t,kappa)
%     r = kappa/100*exp(-((time - .05)^2)/.00025);
    r = 1;
    if t < .1
        r = sin((2*pi*t)/.01);
    end
end