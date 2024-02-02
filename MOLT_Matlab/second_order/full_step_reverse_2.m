% Full Step Reverse 2 considers timesteps
% -1 through 1/2, using them to compute -3/2 (v,A,J) and -2 (x,phi,rho).

% 1
[x1_elec_nm2, x2_elec_nm2] = advance_particle_positions_2D(x1_elec_nm1, x2_elec_nm1, ...
                                                           v1_elec_nm1, v2_elec_nm1, -dt);

% {x,y} at t_{-1}
x1_elec_nm2 = periodic_shift(x1_elec_nm2, x(1), L_x);
x2_elec_nm2 = periodic_shift(x2_elec_nm2, y(1), L_y);

% 2
x1_elec_ave = (x1_elec_nm2 + x1_elec_nm1)/2; % x at t_{-1/2}
x2_elec_ave = (x1_elec_nm2 + x2_elec_nm1)/2; % y at t_{-1/2}

J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                            x1_elec_ave, x2_elec_ave, ...
                            v1_elec_old, v2_elec_old, ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

% J at t_{-3/2}
J1_mesh(:,:,end-2) = J_mesh(:,:,1);
J2_mesh(:,:,end-2) = J_mesh(:,:,2);

% 3
psi_src(:,:) = (1/sigma_1)*(rho_mesh(:,:,end) + rho_mesh(:,:,end-2)); % psi at t_{-1}
A1_src(:,:)  =     sigma_2*(J1_mesh(:,:,end)  + J1_mesh(:,:,end-2)); % A1  at t_{-1/2}
A2_src(:,:)  =     sigma_2*(J1_mesh(:,:,end)  + J2_mesh(:,:,end-2)); % A2  at t_{-1/2}

%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

alpha = beta_CDF2/(kappa*(-dt));

psi_source_with_prev = 2*psi(:,:,end-2) + 1/(alpha^2)*psi_src;
A1_source_with_prev  = 2*A1(:,:,end-1)  + 1/(alpha^2)*A1_src;
A2_source_with_prev  = 2*A2(:,:,end-1)  + 1/(alpha^2)*A2_src;

psi_nm2 = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2) - psi(:,:,end);

A1(:,:,end-2)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A1(:,:,end);

A2(:,:,end-2)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A2(:,:,end);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi_nm2 = compute_ddx_FFT(psi_nm2, kx_deriv_1);
ddy_psi_nm2 = compute_ddy_FFT(psi_nm2, ky_deriv_1);

ddx_A1(:,:,end-2) = compute_ddx_FFT(A1(:,:,end-2), kx_deriv_1);
ddy_A1(:,:,end-2) = compute_ddy_FFT(A1(:,:,end-2), ky_deriv_1);

ddx_A2(:,:,end-2) = compute_ddx_FFT(A2(:,:,end-2), kx_deriv_1);
ddy_A2(:,:,end-2) = compute_ddy_FFT(A2(:,:,end-2), ky_deriv_1);

% 4. Momentum advance by -dt

ddx_psi_ave = (ddx_psi(:,:,end) + ddx_psi(:,:,end-1)) / 2;
ddy_psi_ave = (ddy_psi(:,:,end) + ddy_psi(:,:,end-1)) / 2;

[v1_elec_nm3_2, v2_elec_nm3_2, P1_elec_nm3_2, P2_elec_nm3_2] = ...
improved_asym_euler_momentum_push_2D2P_implicit(x1_elec_nm1, x2_elec_nm1, ...
                                                P1_elec_old, P2_elec_old, ...
                                                v1_elec_old, v2_elec_old, ...
                                                v1_elec_new, v2_elec_new, ...
                                                ddx_psi_ave, ddy_psi_ave, ...
                                                A1(:,:,end), ddx_A1(:,:,end), ddy_A1(:,:,end), ...
                                                A2(:,:,end), ddx_A2(:,:,end), ddy_A2(:,:,end), ...
                                                x, y, dx, dy, q_elec, r_elec, ...
                                                kappa, -dt);