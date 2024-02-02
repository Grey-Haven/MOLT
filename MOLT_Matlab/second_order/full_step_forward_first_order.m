%---------------------------------------------------------------------
% 1. Advance electron positions by dt using v^{n}
%---------------------------------------------------------------------

x1_elec_old = x1_elec_hist(:,end-1);
x2_elec_old = x2_elec_hist(:,end-1);

v1_elec_old = v1_elec_hist(:,end-1);
v2_elec_old = v2_elec_hist(:,end-1);

v1_elec_nm1 = v1_elec_hist(:,end-2);
v2_elec_nm1 = v2_elec_hist(:,end-2);

P1_elec_old = P1_elec_hist(:,end-1);
P2_elec_old = P2_elec_hist(:,end-1);

v1_star = 2*v1_elec_old - v1_elec_nm1;
v2_star = 2*v2_elec_old - v2_elec_nm1;
[x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                           v1_star, v2_star, dt);


% Apply the particle boundary conditions
% Need to include the shift function here
x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);
%---------------------------------------------------------------------

%---------------------------------------------------------------------
% 2. Compute the electron current density used for updating A
%    Compute also the charge density used for updating psi
%---------------------------------------------------------------------
J_rho_update_fft;

%---------------------------------------------------------------------
% 3.1. Compute wave sources
%---------------------------------------------------------------------
psi_src(:,:) = (1/sigma_1)*rho_mesh(:,:,end);
A1_src(:,:) = sigma_2*J1_mesh(:,:,end);
A2_src(:,:) = sigma_2*J2_mesh(:,:,end);

%---------------------------------------------------------------------
% 3.2 Update the scalar (phi) and vector (A) potentials waves. 
%---------------------------------------------------------------------
update_waves_hybrid_FFT;

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
ddt_psi(:,:,end) = ( psi(:,:,end) - psi(:,:,end-1) ) / dt;

% Compute the residual in the Lorenz gauge 
gauge_residual(:,:) = (1/kappa^2)*ddt_psi(:,:,end) + ddx_A1(:,:,end) + ddy_A2(:,:,end);

gauge_error_L2(steps+1) = get_L_2_error(gauge_residual(:,:), ...
                                        zeros(size(gauge_residual(:,:))), ...
                                        dx*dy);
gauge_error_inf(steps+1) = max(max(abs(gauge_residual)));

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