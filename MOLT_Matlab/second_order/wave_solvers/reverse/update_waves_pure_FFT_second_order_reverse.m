%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

alpha = beta_CDF2/(kappa*dt);

beg = 1;

psi_source_with_prev = 2*psi(:,:,beg+1) + 1/(alpha^2)*psi_src;
A1_source_with_prev  = 2*A1(:,:,beg+1)  + 1/(alpha^2)*A1_src;
A2_source_with_prev  = 2*A2(:,:,beg+1)  + 1/(alpha^2)*A2_src;

psi(:,:,beg) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2) - psi(:,:,beg+2);

A1(:,:,beg)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A1(:,:,beg+2);

A2(:,:,beg)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A2(:,:,beg+2);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,beg) = compute_ddx_FFT(psi(:,:,beg), kx_deriv_1);
ddy_psi(:,:,beg) = compute_ddy_FFT(psi(:,:,beg), ky_deriv_1);

ddx_A1(:,:,beg)  = compute_ddx_FFT(A1(:,:,beg) , kx_deriv_1);
ddy_A1(:,:,beg)  = compute_ddy_FFT(A1(:,:,beg) , ky_deriv_1);

ddx_A2(:,:,beg)  = compute_ddx_FFT(A2(:,:,beg) , kx_deriv_1);
ddy_A2(:,:,beg)  = compute_ddy_FFT(A2(:,:,beg) , ky_deriv_1);