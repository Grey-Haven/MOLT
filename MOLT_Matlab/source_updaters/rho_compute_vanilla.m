rho_elec = map_rho_to_mesh_2D(x, y, dx, dy, x1_elec_new, x2_elec_new, ...
                              q_elec, cell_volumes, w_elec);

rho_elec = enforce_periodicity(rho_elec);