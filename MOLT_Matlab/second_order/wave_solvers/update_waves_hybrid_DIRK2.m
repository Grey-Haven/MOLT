%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

ddt_psi_curr = (psi(:,:,end-1) - psi(:,:,end-2)) / dt;
ddt_A1_curr  = (A1(:,:,end-1)  - A1(:,:,end-2))  / dt;
ddt_A2_curr  = (A2(:,:,end-1)  - A2(:,:,end-2))  / dt;

psi_curr = psi(:,:,end-1);
A1_curr  = A1(:,:,end-1);
A2_curr  = A2(:,:,end-1);

psi_src = (1/sigma_1)*rho_mesh;
A1_src  = sigma_2*J1_mesh;
A2_src  = sigma_2*J2_mesh;

[psi_next, ddt_psi_next] = DIRK2_advance_per(psi_curr, ddt_psi_curr, psi_src, kappa, dt, kx_deriv_2, ky_deriv_2);
[A1_next , ddt_A1_next ] = DIRK2_advance_per(A1_curr , ddt_A1_curr , A1_src , kappa, dt, kx_deriv_2, ky_deriv_2);
[A2_next , ddt_A2_next ] = DIRK2_advance_per(A2_curr , ddt_A2_curr , A2_src , kappa, dt, kx_deriv_2, ky_deriv_2);

psi(:,:,end) = psi_next;
A1(:,:,end)  = A1_next;
A2(:,:,end)  = A2_next;

ddt_psi_hist(:,:,end) = ddt_psi_next;
ddt_A1_hist(:,:,end)  = ddt_A1_next;
ddt_A2_hist(:,:,end)  = ddt_A2_next;

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);