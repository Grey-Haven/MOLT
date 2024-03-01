%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

ddt_psi = (psi(:,:,end-1) - psi(:,:,end-2)) / dt;
ddt_A1  = (A1(:,:,end-1)  - A1(:,:,end-2) ) / dt;
ddt_A2  = (A2(:,:,end-1)  - A2(:,:,end-2) ) / dt;

psi_curr = psi(:,:,end-1);
A1_curr  = A1(:,:,end-1);
A2_curr  = A2(:,:,end-1);

[psi_next, ddt_psi_next] = DIRK2_advance_per(psi_curr, ddt_psi, psi_src_hist, kappa, dt, kx_deriv_2, ky_deriv_2);
[A1_next , ddt_A1_next ] = DIRK2_advance_per(A1_curr , ddt_A1,  A1_src_hist , kappa, dt, kx_deriv_2, ky_deriv_2);
[A2_next , ddt_A2_next ] = DIRK2_advance_per(A2_curr , ddt_A2,  A2_src_hist , kappa, dt, kx_deriv_2, ky_deriv_2);

psi(:,:,end) = psi_next;
A1(:,:,end)  = A1_next;
A2(:,:,end)  = A2_next;

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);