%---------------------------------------------------------------------
% 5.2.1. Advance psi by dt using BDF-1 
%---------------------------------------------------------------------

ik = 1i*[0:N/2 -N/2+1:-1]; % i * wave number vector (matlab ordering)
ik2 = ik.*ik;

laplacian_hat = kappa^2*dt^2*(-kx_deriv_2'.^2 - ky_deriv_2.^2);
split_term_hat = kappa^4*dt^4*kx_deriv_2'.^2 .* ky_deriv_2.^2;

psi_prev = psi(1:end-1,1:end-1,1);
psi_curr = psi(1:end-1,1:end-1,2);
psi_src_curr = psi_src(1:end-1,1:end-1);

psi_prev_hat = fft2(psi_prev);
psi_curr_hat = fft2(psi_curr);
psi_src_hat = fft2(psi_src_curr);

psi_next_hat = (2*psi_curr_hat - psi_prev_hat + kappa^2*dt^2*psi_src_hat) ./ (1 - laplacian_hat + split_term_hat);

psi_next = ifft2(psi_next_hat);

psi(1:end-1,1:end-1,3) = psi_next;
psi(:,:,3) = copy_periodic_boundaries(psi(:,:,3));

%---------------------------------------------------------------------
% 5.2.2. Advance A1 by dt using BDF-1
%---------------------------------------------------------------------

A1_prev = A1(1:end-1,1:end-1,1);
A1_curr = A1(1:end-1,1:end-1,2);
A1_src_curr = A1_src(1:end-1,1:end-1);

A1_prev_hat = fft2(A1_prev);
A1_curr_hat = fft2(A1_curr);
A1_src_hat = fft2(A1_src_curr);

A1_next_hat = (2*A1_curr_hat - A1_prev_hat + kappa^2*dt^2*A1_src_hat) ./ (1 - laplacian_hat + split_term_hat);

A1_next = ifft2(A1_next_hat);

A1(1:end-1,1:end-1,3) = A1_next;
A1(:,:,3) = copy_periodic_boundaries(A1(:,:,3));

%---------------------------------------------------------------------
% 5.2.3. Advance A2 by dt using BDF-1
%---------------------------------------------------------------------

A2_prev = A2(1:end-1,1:end-1,1);
A2_curr = A2(1:end-1,1:end-1,2);
A2_src_curr = A2_src(1:end-1,1:end-1);

A2_prev_hat = fft2(A2_prev);
A2_curr_hat = fft2(A2_curr);
A2_src_hat = fft2(A2_src_curr);

A2_next_hat = (2*A2_curr_hat - A2_prev_hat + kappa^2*dt^2*A2_src_hat) ./ (1 - laplacian_hat + split_term_hat);

A2_next = ifft2(A2_next_hat);

A2(1:end-1,1:end-1,3) = A2_next;
A2(:,:,3) = copy_periodic_boundaries(A2(:,:,3));

%---------------------------------------------------------------------
% 5.2.4. Compute the derivatives of the above fields
%---------------------------------------------------------------------

% Psi
psi_next_fft_x = fft(psi_next,N_x-1,2);
psi_next_fft_y = fft(psi_next,N_y-1,1);

ddx_psi(1:end-1,1:end-1) = real(ifft(sqrt(-1)*kx_deriv_1 .*psi_next_fft_x,N_x-1,2));
ddy_psi(1:end-1,1:end-1) = real(ifft(sqrt(-1)*ky_deriv_1'.*psi_next_fft_y,N_y-1,1));

ddx_psi = copy_periodic_boundaries(ddx_psi);
ddy_psi = copy_periodic_boundaries(ddy_psi);

% A1
A1_next_fft_x = fft(A1_next,N_x-1,2);
A1_next_fft_y = fft(A1_next,N_y-1,1);

ddx_A1(1:end-1,1:end-1) = real(ifft(sqrt(-1)*kx_deriv_1 .*A1_next_fft_x,N_x-1,2));
ddy_A1(1:end-1,1:end-1) = real(ifft(sqrt(-1)*ky_deriv_1'.*A1_next_fft_y,N_y-1,1));

ddx_A1 = copy_periodic_boundaries(ddx_A1);
ddy_A1 = copy_periodic_boundaries(ddy_A1);

% A2
A2_next_fft_x = fft(A2_next,N_x-1,2);
A2_next_fft_y = fft(A2_next,N_y-1,1);

ddx_A2(1:end-1,1:end-1) = real(ifft(sqrt(-1)*kx_deriv_1 .*A2_next_fft_x,N_x-1,2));
ddy_A2(1:end-1,1:end-1) = real(ifft(sqrt(-1)*ky_deriv_1'.*A2_next_fft_y,N_y-1,1));

ddx_A2 = copy_periodic_boundaries(ddx_A2);
ddy_A2 = copy_periodic_boundaries(ddy_A2);

assert(norm(imag(ddx_psi)) < eps);
assert(norm(imag(ddy_psi)) < eps);
assert(norm(imag(ddx_A1)) < eps);
assert(norm(imag(ddy_A1)) < eps);
assert(norm(imag(ddx_A2)) < eps);
assert(norm(imag(ddy_A2)) < eps);