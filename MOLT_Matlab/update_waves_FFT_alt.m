%---------------------------------------------------------------------
% 5.2.1. Advance psi by dt using BDF-1 
%---------------------------------------------------------------------


psi_prev = psi(1:end-1,1:end-1,1);
psi_curr = psi(1:end-1,1:end-1,2);
psi_src_curr = psi_src(1:end-1,1:end-1);

psi_prev_hat = fft2(psi_prev);
psi_curr_hat = fft2(psi_curr);
psi_src_hat = fft2(psi_src_curr);

psi_next_hat = (2*psi_curr_hat - psi_prev_hat + kappa^2*dt^2*psi_src_hat) ./ (1 - kappa^2*dt^2*(-kx'.^2 - ky.^2));

% psi_next = ifft(ifft(psi_next_hat,N_y-1,1),N_x-1,2);
psi_next = ifft2(psi_next_hat);

% psi_xx = ifft(-(kx.^2) .*psi_curr_fft_x,N_x-1,2);
% psi_yy = ifft(-(ky.^2)'.*psi_curr_fft_y,N_y-1,1);
% 
% psi_tilde = psi_xx + psi_yy;
% 
% psi_next = 2*psi_curr - psi_prev - kappa^2*dt^2*(psi_src(1:end-1,1:end-1) - psi_tilde);

psi(1:end-1,1:end-1,3) = psi_next;
psi(:,:,3) = copy_periodic_boundaries(psi(:,:,3));

%---------------------------------------------------------------------
% 5.2.2. Advance A1 by dt using BDF-1
%---------------------------------------------------------------------

A1_prev = A1(1:end-1,1:end-1,1);
A1_curr = A1(1:end-1,1:end-1,2);

A1_fft_x = fft(A1_curr,N_x-1,2);
A1_fft_y = fft(A1_curr,N_y-1,1);

A1_xx = ifft(-(kx.^2) .*A1_fft_x,N_x-1,2);
A1_yy = ifft(-(ky.^2)'.*A1_fft_y,N_y-1,1);

A1_tilde = A1_xx + A1_yy;
A1_next = 2*A1_curr - A1_prev - kappa^2*dt^2*(A1_src(1:end-1,1:end-1) - A1_tilde);

A1(1:end-1,1:end-1,3) = A1_next;
A1(:,:,3) = copy_periodic_boundaries(A1(:,:,3));

%---------------------------------------------------------------------
% 5.2.3. Advance A2 by dt using BDF-1
%---------------------------------------------------------------------

A2_prev = A2(1:end-1,1:end-1,1);
A2_curr = A2(1:end-1,1:end-1,2);

A2_fft_x = fft(A2_curr,N_x-1,2);
A2_fft_y = fft(A2_curr,N_y-1,1);

A2_xx = ifft(-(kx.^2) .*A2_fft_x,N_x-1,2);
A2_yy = ifft(-(ky.^2)'.*A2_fft_y,N_y-1,1);

A2_tilde = A2_xx + A2_yy;
A2_next = 2*A2_curr - A2_prev - kappa^2*dt^2*(A2_src(1:end-1,1:end-1) - A2_tilde);

A2(1:end-1,1:end-1,3) = A2_next;
A2(:,:,3) = copy_periodic_boundaries(A2(:,:,3));

%---------------------------------------------------------------------
% 5.2.4. Compute the derivatives of the above fields
%---------------------------------------------------------------------

% Psi
psi_next_fft_x = fft(psi_next,N_x-1,2);
psi_next_fft_y = fft(psi_next,N_y-1,1);

ddx_psi(1:end-1,1:end-1) = ifft(sqrt(-1)*kx .*psi_next_fft_x,N_x-1,2);
ddy_psi(1:end-1,1:end-1) = ifft(sqrt(-1)*ky'.*psi_next_fft_y,N_y-1,1);

ddx_psi = copy_periodic_boundaries(ddx_psi);
ddy_psi = copy_periodic_boundaries(ddy_psi);

% A1
A1_next_fft_x = fft(A1_next,N_x-1,2);
A1_next_fft_y = fft(A1_next,N_y-1,1);

ddx_A1(1:end-1,1:end-1) = ifft(sqrt(-1)*kx .*A1_next_fft_x,N_x-1,2);
ddy_A1(1:end-1,1:end-1) = ifft(sqrt(-1)*ky'.*A1_next_fft_y,N_y-1,1);

ddx_A1 = copy_periodic_boundaries(ddx_A1);
ddy_A1 = copy_periodic_boundaries(ddy_A1);

% A2
A2_next_fft_x = fft(A2_next,N_x-1,2);
A2_next_fft_y = fft(A2_next,N_y-1,1);

ddx_A2(1:end-1,1:end-1) = ifft(sqrt(-1)*kx .*A2_next_fft_x,N_x-1,2);
ddy_A2(1:end-1,1:end-1) = ifft(sqrt(-1)*ky'.*A2_next_fft_y,N_y-1,1);

ddx_A2 = copy_periodic_boundaries(ddx_A2);
ddy_A2 = copy_periodic_boundaries(ddy_A2);

