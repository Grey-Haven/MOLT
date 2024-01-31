%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

psi(:,:,end) = BDF2_implicit_advance_per(psi, psi_src(:,:), x, y, dx, dy, dt, kappa, beta_BDF);

A1(:,:,end)  = BDF2_implicit_advance_per(A1 , A1_src(:,:) , x, y, dx, dy, dt, kappa, beta_BDF);

A2(:,:,end)  = BDF2_implicit_advance_per(A1 , A2_src(:,:) , x, y, dx, dy, dt, kappa, beta_BDF);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);