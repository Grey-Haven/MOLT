% Compute the next step of rho using the continuity equation.
% The FFT will be used to compute div(J).

J_compute_vanilla;

ddx_J1_FD6 = zeros(size(J1_mesh));
ddy_J2_FD6 = zeros(size(J2_mesh));

Nx = N_x - 1;
Ny = N_y - 1;

for i = 1:Nx
    i_idx_m3 = i-3;
    i_idx_m2 = i-2;
    i_idx_m1 = i-1;
    i_idx    = i+0;
    i_idx_p1 = i+1;
    i_idx_p2 = i+2;
    i_idx_p3 = i+3;
    i_idx_m3 = mod(i_idx_m3 - 1, Nx) + 1;
    i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
    i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
    i_idx    = mod(i_idx    - 1, Nx) + 1;
    i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
    i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
    i_idx_p3 = mod(i_idx_p3 - 1, Nx) + 1;
    for j = 1:Ny
        j_idx_m3 = j-3;
        j_idx_m2 = j-2;
        j_idx_m1 = j-1;
        j_idx    = j+0;
        j_idx_p1 = j+1;
        j_idx_p2 = j+2;
        j_idx_p3 = j+3;
        j_idx_m3 = mod(j_idx_m3 - 1, Nx) + 1;
        j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
        j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
        j_idx    = mod(j_idx    - 1, Ny) + 1;
        j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
        j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
        j_idx_p3 = mod(j_idx_p3 - 1, Nx) + 1;

        ddx_J1_FD6(j_idx,i_idx) = (-1/60*J1_mesh(j_idx,i_idx_m3) + 3/20*J1_mesh(j_idx,i_idx_m2) - 3/4*J1_mesh(j_idx,i_idx_m1) + 3/4*J1_mesh(j_idx,i_idx_p1) - 3/20*J1_mesh(j_idx,i_idx_p2) + 1/60*J1_mesh(j_idx,i_idx_p3)) / (dx);
        ddy_J2_FD6(j_idx,i_idx) = (-1/60*J2_mesh(j_idx_m3,i_idx) + 3/20*J2_mesh(j_idx_m2,i_idx) - 3/4*J2_mesh(j_idx_m1,i_idx) + 3/4*J2_mesh(j_idx_p1,i_idx) - 3/20*J2_mesh(j_idx_p2,i_idx) + 1/60*J2_mesh(j_idx_p3,i_idx)) / (dy);
    end
end

ddx_J1_FD6 = copy_periodic_boundaries(ddx_J1_FD6);
ddy_J2_FD6 = copy_periodic_boundaries(ddy_J2_FD6);

J1_star_deriv = ddx_J1_FD6(1:end-1,1:end-1);
J2_star_deriv = ddy_J2_FD6(1:end-1,1:end-1);

rho_mesh(1:end-1,1:end-1) = rho_mesh(1:end-1,1:end-1) - dt*(J1_star_deriv + J2_star_deriv);
rho_mesh(end,:) = rho_mesh(1,:);
rho_mesh(:,end) = rho_mesh(:,1);