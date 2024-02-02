% 1
v1_star = 2*v1_elec_old - v1_elec_nm1;
v2_star = 2*v2_elec_old - v2_elec_nm1;
[x1_elec_new, x2_elec_new] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                           v1_star, v2_star, dt);

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



% 3
psi_src(:,:) = (1/sigma_1)*(rho_mesh(:,:,end) + rho_mesh(:,:,end-2));
A1_src(:,:)  =     sigma_2*(J1_mesh(:,:,end)  + J1_mesh(:,:,end-2) );
A2_src(:,:)  =     sigma_2*(J2_mesh(:,:,end)  + J2_mesh(:,:,end-2) );
update_waves_pure_FFT_second_order;

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