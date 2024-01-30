% A1 uses J1
[A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance_hybrid_FFT(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);

% A2 uses J2
[A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance_hybrid_FFT(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);


div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
ddt_div_A = (div_A_curr - div_A_prev)/dt;

RHS = -(rho_mesh / sigma_1 + ddt_div_A);
LHS = solve_poisson_FFT(RHS,kx_deriv_2,ky_deriv_2);
psi(:,:,end) = LHS;

psi_next_fft_x = fft(psi(1:end-1,1:end-1,end),N_x-1,2);
psi_next_fft_y = fft(psi(1:end-1,1:end-1,end),N_y-1,1);

ddx_psi_fft = zeros(N_y,N_x);
ddy_psi_fft = zeros(N_y,N_x);

ddx_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*psi_next_fft_x,N_x-1,2);
ddy_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*psi_next_fft_y,N_y-1,1);

ddx_psi_fft = copy_periodic_boundaries(ddx_psi_fft);
ddy_psi_fft = copy_periodic_boundaries(ddy_psi_fft);

ddx_psi(:,:,end) = ddx_psi_fft;
ddy_psi(:,:,end) = ddy_psi_fft;