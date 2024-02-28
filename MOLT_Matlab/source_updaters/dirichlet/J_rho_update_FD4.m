% Compute the next step of rho using the continuity equation.
% The FD2 will be used to compute div(J).

J_compute_vanilla;

[ddx_J1_FD4, ddy_J1_FD4__] = compute_gradient_FD4_dir(J1_mesh(:,:,end),dx,dy);
[ddx_J2_FD4__, ddy_J2_FD4] = compute_gradient_FD4_dir(J2_mesh(:,:,end),dx,dy);

rho_mesh(:,:,end) = (4/3)*rho_mesh(:,:,end-1) - (1/3)*rho_mesh(:,:,end-2) - (2/3)*dt*(ddx_J1_FD4 + ddy_J2_FD4);
rho_mesh(:,:,end) = enforce_dirichlet(rho_mesh(:,:,end),0,0,0,0);
% rho_mesh(:,:,end) = rho_mesh(:,:,end-1) - dt*(ddx_J1_FD2 + ddy_J2_FD2);