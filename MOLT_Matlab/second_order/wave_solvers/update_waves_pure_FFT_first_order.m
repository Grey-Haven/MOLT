%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

alpha = beta_BDF/(kappa*dt);

psi_source_with_prev = 2*psi(:,:,end-1) - psi(:,:,end-2) + 1/(alpha^2)*psi_src;
A1_source_with_prev  = 2*A1(:,:,end-1)  - A1(:,:,end-2)  + 1/(alpha^2)*A1_src;
A2_source_with_prev  = 2*A2(:,:,end-1)  - A2(:,:,end-2)  + 1/(alpha^2)*A2_src;

psi(:,:,end) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2);

A1(:,:,end)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);

A2(:,:,end)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);