rho_elec = map_rho_to_mesh_2D(x, y, dx, dy, x1_elec_new, x2_elec_new, ...
                              q_elec, cell_volumes, w_elec);

rho_elec = enforce_periodicity(rho_elec);

rho_elec(1:end-1,1:end-1,1) = ifft(ifft(fft(fft(rho_elec(1:end-1,1:end-1,1),N_x-1,1),N_y-1,2),N_x-1,1),N_y-1,2);

% Need to enforce periodicity for the charge on the mesh
rho_elec(end,:) = rho_elec(1,:);
rho_elec(:,end) = rho_elec(:,1);