% Assuming A, J, v at integer timesteps
% Assuming rho, phi, x at half timesteps

x1_elec_ave = (x1_elec_new + x1_elec_old)/2;
x2_elec_ave = (x2_elec_new + x2_elec_old)/2;

J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                            x1_elec_ave, x2_elec_ave, ...
                            v1_star, v2_star, ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J1_mesh(:,:,end) = enforce_periodicity(J_mesh(:,:,1));
J2_mesh(:,:,end) = enforce_periodicity(J_mesh(:,:,2));

rho_compute_vanilla;

rho_mesh(:,:,end) = rho_ions(:,:) + rho_elec(:,:);