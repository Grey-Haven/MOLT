psi_src = 1/sigma_1 * rho_mesh(:,:,end);
A1_src  =   sigma_2 *  J1_mesh(:,:,end);
A2_src  =   sigma_2 *  J2_mesh(:,:,end);

%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

psi(:,:,end) = BDF1_per_advance(psi, psi_src, x, y, dx, dy, dt, kappa, beta_BDF1);

A1(:,:,end)  = BDF1_per_advance(A1 , A1_src , x, y, dx, dy, dt, kappa, beta_BDF1);

A2(:,:,end)  = BDF1_per_advance(A2 , A2_src , x, y, dx, dy, dt, kappa, beta_BDF1);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);