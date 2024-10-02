psi_src = 1/sigma_1 * rho_mesh(:,:,end);
A1_src  =   sigma_2 *  J1_mesh(:,:,end);
A2_src  =   sigma_2 *  J2_mesh(:,:,end);

%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

psi(:,:,end) = BDF2_per_advance(psi, psi_src, x, y, dx, dy, dt, kappa, beta_BDF2);

A1(:,:,end)  = BDF2_per_advance(A1 , A1_src , x, y, dx, dy, dt, kappa, beta_BDF2);

A2(:,:,end)  = BDF2_per_advance(A2 , A2_src , x, y, dx, dy, dt, kappa, beta_BDF2);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FD6
%---------------------------------------------------------------------

ddx_psi(1:end-1,1:end-1,end) = compute_ddx_FD6_per(psi(1:end-1,1:end-1,end), dx);
ddy_psi(1:end-1,1:end-1,end) = compute_ddy_FD6_per(psi(1:end-1,1:end-1,end), dy);

ddx_A1(1:end-1,1:end-1,end)  = compute_ddx_FD6_per(A1(1:end-1,1:end-1,end) , dx);
ddy_A1(1:end-1,1:end-1,end)  = compute_ddy_FD6_per(A1(1:end-1,1:end-1,end) , dy);

ddx_A2(1:end-1,1:end-1,end)  = compute_ddx_FD6_per(A2(1:end-1,1:end-1,end) , dx);
ddy_A2(1:end-1,1:end-1,end)  = compute_ddy_FD6_per(A2(1:end-1,1:end-1,end) , dy);

ddx_psi(:,:,end) = copy_periodic_boundaries(ddx_psi(:,:,end));
ddy_psi(:,:,end) = copy_periodic_boundaries(ddy_psi(:,:,end));

ddx_A1(:,:,end) = copy_periodic_boundaries(ddx_A1(:,:,end));
ddy_A1(:,:,end) = copy_periodic_boundaries(ddy_A1(:,:,end));

ddx_A2(:,:,end) = copy_periodic_boundaries(ddx_A2(:,:,end));
ddy_A2(:,:,end) = copy_periodic_boundaries(ddy_A2(:,:,end));