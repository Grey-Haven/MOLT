% Compute the next step of rho using the continuity equation.
% The FFT will be used to compute div(J).

% foo = sin(6*pi*y)' .* sin(4*pi*x);
% foo_fft_x = fft(foo(1:end-1,1:end-1),N_x-1,2);
% foo_deriv_x = ifft(sqrt(-1)*kx_old_f .* foo_fft_x,N_x-1,2);
% foo_analytic_deriv_x = 4*pi* sin(6*pi*y)' .* cos(4*pi*x);

J_compute_vanilla;

J1_clean = ifft(fft(ifft(fft(J1_mesh(1:end-1,1:end-1),N_x-1,2),N_x-1,2),N_y-1,1),N_y-1,1);
J2_clean = ifft(fft(ifft(fft(J2_mesh(1:end-1,1:end-1),N_x-1,2),N_x-1,2),N_y-1,1),N_y-1,1);

J1_clean_FFTx = fft(J1_clean,N_x-1,2);
J2_clean_FFTy = fft(J2_clean,N_y-1,1);

J1_deriv_clean = ifft(sqrt(-1)*kx_deriv_1 .*J1_clean_FFTx,N_x-1,2);
J2_deriv_clean = ifft(sqrt(-1)*ky_deriv_1'.*J2_clean_FFTy,N_y-1,1);

Gamma = -1/((N_x-1)*(N_y-1))*sum(sum(J1_deriv_clean + J2_deriv_clean));

F1 = .5*Gamma*x(1:end-1)'.*ones(N_y-1,N_x-1);
F2 = .5*Gamma*y(1:end-1) .*ones(N_y-1,N_x-1);

J1_star = J1_clean + F1;
J2_star = J2_clean + F2;

J1_star_FFTx = fft(J1_star,N_x-1,2);
J2_star_FFTy = fft(J2_star,N_y-1,1);

J1_star_deriv = ifft(sqrt(-1)*kx_deriv_1 .*J1_star_FFTx,N_x-1,2);
J2_star_deriv = ifft(sqrt(-1)*ky_deriv_1'.*J2_star_FFTy,N_y-1,1);

% J1_deriv_clean(1:end-1,1:end-1) = J1_star_deriv;
% J2_deriv_clean(1:end-1,1:end-1) = J2_star_deriv;

rho_mesh(1:end-1,1:end-1) = rho_mesh(1:end-1,1:end-1) - dt*(J1_star_deriv + J2_star_deriv);
rho_mesh(end,:) = rho_mesh(1,:);
rho_mesh(:,end) = rho_mesh(:,1);

J1_mesh(1:end-1,1:end-1) = J1_star;
J2_mesh(1:end-1,1:end-1) = J2_star;
J1_mesh(end,:) = J1_mesh(1,:);
J1_mesh(:,end) = J1_mesh(:,1);
J2_mesh(end,:) = J2_mesh(1,:);
J2_mesh(:,end) = J2_mesh(:,1);

J_mesh(:,:,1) = J1_mesh;
J_mesh(:,:,2) = J2_mesh;