% Compute the next step of rho using the continuity equation.
% The FD2 will be used to compute div(J).

J_compute_vanilla;

ddx_J1_FD2 = zeros(size(J1_mesh(:,:,end)));
ddy_J2_FD2 = zeros(size(J2_mesh(:,:,end)));

for i = 2:N_x-1
    for j = 2:N_y-1
        ddx_J1_FD2(j, i) = (J1_mesh(j, i+1, end) - J1_mesh(j, i-1, end)) / (2*dx);
        ddy_J2_FD2(j, i) = (J2_mesh(j+1, i, end) - J2_mesh(j-1, i, end)) / (2*dy);
    end
end

% rho_mesh(:,:,end) = 0.5*( 5*rho_mesh(:,:,end-1) - 4*rho_mesh(:,:,end-2) + rho_mesh(:,:,end-3) ) - dt*(ddx_J1_FD2 + ddy_J2_FD2);
rho_mesh(:,:,end) = rho_mesh(:,:,end-1) - dt*(ddx_J1_FD2 + ddy_J2_FD2);