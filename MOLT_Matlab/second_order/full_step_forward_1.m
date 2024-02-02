% Full Step Forward 1 considers timesteps
% -3/2 through 0, using them to compute 1/2 (v,A,J) and 1 (x,phi,rho).

% 1
[x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                           v1_elec_new, v2_elec_new, dt);

x1_elec_new = periodic_shift(x1_elec_new, x(1), L_x);
x2_elec_new = periodic_shift(x2_elec_new, y(1), L_y);

% 2
x1_elec_ave = (x1_elec_new + x1_elec_old)/2;
x2_elec_ave = (x2_elec_new + x2_elec_old)/2;

J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                            x1_elec_ave, x2_elec_ave, ...
                            v1_elec_old, v2_elec_old, ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

J1_mesh(:,:,end) = J_mesh(:,:,1);
J2_mesh(:,:,end) = J_mesh(:,:,2);

J1_deriv_x = compute_ddx_FFT(J1_mesh(:,:,end), kx_deriv_1);
J2_deriv_y = compute_ddy_FFT(J2_mesh(:,:,end), ky_deriv_1);

rho_mesh(:,:,end) = rho_mesh(:,:,end-1) - dt*(J1_deriv_x + J2_deriv_y);

% 3

% 3.1 Compute the sources
psi_src(:,:) = (1/sigma_1)*(rho_mesh(:,:,end) + rho_mesh(:,:,end-2));
A1_src(:,:)  =     sigma_2*(J1_mesh(:,:,end)  + J1_mesh(:,:,end-2) );
A2_src(:,:)  =     sigma_2*(J2_mesh(:,:,end)  + J2_mesh(:,:,end-2) );

alpha = beta_CDF2/(kappa*dt);

psi_source_with_prev = 2*psi(:,:,end-1) + 1/(alpha^2)*psi_src;
A1_source_with_prev  = 2*A1(:,:,end-1)  + 1/(alpha^2)*A1_src;
A2_source_with_prev  = 2*A2(:,:,end-1)  + 1/(alpha^2)*A2_src;

% 3.2 Advance the waves
psi(:,:,end) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2) - psi(:,:,end-2);

A1(:,:,end)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A1(:,:,end-2);

A2(:,:,end)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A2(:,:,end-2);

% 3.3. Compute their derivatives
ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end) = compute_ddx_FFT(A1(:,:,end), kx_deriv_1);
ddy_A1(:,:,end) = compute_ddy_FFT(A1(:,:,end), ky_deriv_1);

ddx_A2(:,:,end) = compute_ddx_FFT(A2(:,:,end), kx_deriv_1);
ddy_A2(:,:,end) = compute_ddy_FFT(A2(:,:,end), ky_deriv_1);

% 4. Momentum advance by dt

ddx_psi_ave = (ddx_psi(:,:,end) + ddx_psi(:,:,end-1)) / 2;
ddy_psi_ave = (ddy_psi(:,:,end) + ddy_psi(:,:,end-1)) / 2;

[v1_elec_new, v2_elec_new, P1_elec_new, P2_elec_new] = ...
improved_asym_euler_momentum_push_2D2P_implicit(x1_elec_new, x2_elec_new, ...
                                                P1_elec_old, P2_elec_old, ...
                                                v1_elec_old, v2_elec_old, ...
                                                v1_elec_nm1, v2_elec_nm1, ...
                                                ddx_psi_ave, ddy_psi_ave, ...
                                                A1(:,:,end), ddx_A1(:,:,end), ddy_A1(:,:,end), ...
                                                A2(:,:,end), ddx_A2(:,:,end), ddy_A2(:,:,end), ...
                                                x, y, dx, dy, q_elec, r_elec, ...
                                                kappa, dt);