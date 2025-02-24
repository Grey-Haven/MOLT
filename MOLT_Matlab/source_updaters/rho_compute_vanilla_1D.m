rho_elec = map_rho_to_mesh_1D(x, dx, x1_elec_new, q_elec, cell_volumes, w_elec);

rho_elec = enforce_periodicity_1D(rho_elec);
rho_mesh = rho_elec + rho_ions;