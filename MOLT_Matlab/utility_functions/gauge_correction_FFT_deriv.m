div_A = ddx_A1 + ddy_A2;

psi_prev = psi(:,:,end-1);
psi_A = psi(:,:,end);

psi_C = psi_prev - psi_A - kappa^2*dt*div_A;

psi(:,:,end) = psi_C + psi_A;

psi_next = psi(1:end-1,1:end-1,end);

psi_next_fft_x = fft(psi_next,N_x-1,2);
psi_next_fft_y = fft(psi_next,N_y-1,1);

ddx_psi(1:end-1,1:end-1) = real(ifft(sqrt(-1)*kx_deriv_1 .*psi_next_fft_x,N_x-1,2));
ddy_psi(1:end-1,1:end-1) = real(ifft(sqrt(-1)*ky_deriv_1'.*psi_next_fft_y,N_y-1,1));

ddx_psi = copy_periodic_boundaries(ddx_psi);
ddy_psi = copy_periodic_boundaries(ddy_psi);