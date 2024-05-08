% Map for electrons (ions are stationary)
% Can try using the starred velocities here if we want
% Returns a 3-d array, the first two indices being the y,x coordinates
% and the last being J1 and J2, respectively.
J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                            x1_elec_new, x2_elec_new, ...
                            v1_elec_old, v2_elec_old, ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_dirichlet(J_mesh(:,:,1),0,0,0,0);
J_mesh(:,:,2) = enforce_dirichlet(J_mesh(:,:,2),0,0,0,0);

J1_mesh(:,:,end) = J_mesh(:,:,1);
J2_mesh(:,:,end) = J_mesh(:,:,2);