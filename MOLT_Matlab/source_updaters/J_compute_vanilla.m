% Map for electrons (ions are stationary)
% Can try using the starred velocities here if we want
J_mesh = map_J_to_mesh_2D2V(J_mesh(:,:,:), x, y, dx, dy, ...
                            x1_elec_new, x2_elec_new, ...
                            v1_elec_old, v2_elec_old, ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

assert(all(J_mesh(1,:,1) == J_mesh(end,:,1)));
assert(all(J_mesh(:,1,1) == J_mesh(:,end,1)));

assert(all(J_mesh(1,:,2) == J_mesh(end,:,2)));
assert(all(J_mesh(:,1,2) == J_mesh(:,end,2)));

J1_mesh = J_mesh(:,:,1);
J2_mesh = J_mesh(:,:,2);