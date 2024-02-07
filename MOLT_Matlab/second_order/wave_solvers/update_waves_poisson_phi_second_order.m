% A1 uses J1
[A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance_hybrid_FFT(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);

% A2 uses J2
[A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance_hybrid_FFT(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);


div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
ddt_div_A = (div_A_curr - div_A_prev)/dt;

rho_avg = (rho_mesh(:,:,end) + rho_mesh(:,:,end-2)) / 2;

% Gauge Condition
%    (1/c^2) ddt_phi  + div_A = 0
% => (1/c^2) ddt2_phi + ddt_div_A = 0
% Original Scalar Wave Equation
%    ddt2_phi - laplacian((phi^{n+1} + phi^{n-1})/2) = (rho^{n+1} +
%    rho^{n-1})/(2*sigma_1)
% => -ddt_div_A - laplacian((phi^{n+1} + phi^{n-1})/2) = (rho^{n+1} +
%    rho^{n-1})/(2*sigma_1)
% => laplacian((phi^{n+1} + phi^{n-1})/2) = -(ddt_div_A + (rho^{n+1} +
%    rho^{n-1})/(2*sigma_1))
% => phi^{n+1} = 2*laplacian^{-1}(-(ddt_div_A + (rho^{n+1} +
%    rho^{n-1})/(2*sigma_1))) - phi^{n-1}

RHS = -(rho_avg / sigma_1 + ddt_div_A);
LHS = 2*solve_poisson_FFT(RHS,kx_deriv_2,ky_deriv_2) - psi(:,:,end-2);
psi(:,:,end) = LHS;

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end),kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end),ky_deriv_1);

% psi_next_fft_x = fft(psi(1:end-1,1:end-1,end),N_x-1,2);
% psi_next_fft_y = fft(psi(1:end-1,1:end-1,end),N_y-1,1);
% 
% ddx_psi_fft = zeros(N_y,N_x);
% ddy_psi_fft = zeros(N_y,N_x);
% 
% ddx_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*psi_next_fft_x,N_x-1,2);
% ddy_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*psi_next_fft_y,N_y-1,1);
% 
% ddx_psi_fft = copy_periodic_boundaries(ddx_psi_fft);
% ddy_psi_fft = copy_periodic_boundaries(ddy_psi_fft);
% 
% ddx_psi(:,:,end) = ddx_psi_fft;
% ddy_psi(:,:,end) = ddy_psi_fft;