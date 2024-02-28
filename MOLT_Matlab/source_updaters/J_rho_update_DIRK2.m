% Compute the next step of rho using the continuity equation.
% The FFT will be used to compute div(J).

J_compute_vanilla;

J1_clean = ifft(fft(ifft(fft(J1_mesh(1:end-1,1:end-1,end),N_x-1,2),N_x-1,2),N_y-1,1),N_y-1,1);
J2_clean = ifft(fft(ifft(fft(J2_mesh(1:end-1,1:end-1,end),N_x-1,2),N_x-1,2),N_y-1,1),N_y-1,1);

J1_clean_FFTx = fft(J1_clean,N_x-1,2);
J2_clean_FFTy = fft(J2_clean,N_y-1,1);

J1_deriv_clean = ifft(sqrt(-1)*kx_deriv_1 .*J1_clean_FFTx,N_x-1,2);
J2_deriv_clean = ifft(sqrt(-1)*ky_deriv_1'.*J2_clean_FFTy,N_y-1,1);

Gamma = -1/((N_x-1)*(N_y-1))*sum(sum(J1_deriv_clean + J2_deriv_clean));

F1 = .5*Gamma*x(1:end-1)'.*ones(N_y-1,N_x-1);
F2 = .5*Gamma*y(1:end-1) .*ones(N_y-1,N_x-1);

J1_star = J1_clean + F1;
J2_star = J2_clean + F2;

J1_mesh(1:end-1,1:end-1,end) = J1_star;
J2_mesh(1:end-1,1:end-1,end) = J2_star;

J1_mesh(:,:,end) = copy_periodic_boundaries(J1_mesh(:,:,end));
J2_mesh(:,:,end) = copy_periodic_boundaries(J2_mesh(:,:,end));

ddx_J1_prev = compute_ddx_FFT(J1_mesh(:,:,end-1),kx_deriv_1);
ddy_J2_prev = compute_ddy_FFT(J2_mesh(:,:,end-1),ky_deriv_1);
div_J_prev  = ddx_J1_prev + ddy_J2_prev; 

ddx_J1_curr = compute_ddx_FFT(J1_mesh(:,:,end),kx_deriv_1);
ddy_J2_curr = compute_ddy_FFT(J2_mesh(:,:,end),ky_deriv_1);
div_J_curr  = ddx_J1_curr + ddy_J2_curr; 

% J1_star_FFTx = fft(J1_star,N_x-1,2);
% J2_star_FFTy = fft(J2_star,N_y-1,1);
% 
% J1_star_deriv = ifft(sqrt(-1)*kx_deriv_1 .*J1_star_FFTx,N_x-1,2);
% J2_star_deriv = ifft(sqrt(-1)*ky_deriv_1'.*J2_star_FFTy,N_y-1,1);

div_J_hist = zeros([N_y, N_x, 2]);
div_J_hist(:,:,end-1) = div_J_prev;
div_J_hist(:,:,end)   = div_J_curr;

rho_mesh_nm1  = rho_mesh(:,:,end-2);
rho_mesh_prev = rho_mesh(:,:,end-1);
ddt_rho = (rho_mesh_prev - rho_mesh_nm1) / dt;

% [rho_mesh_next, ddt_rho_mesh_next] = DIRK2_advance_per(rho_mesh(:,:,end-1), ddt_rho, -div_J_hist, kappa, dt, kx_deriv_2, ky_deriv_2);
% 
% rho_mesh(:,:,end) = copy_periodic_boundaries(rho_mesh_next);

b1 = 1/2;
b2 = 1/2;

rho_1 = -((1 - 1/4)*div_J_prev + (1/4)*div_J_curr);
rho_2 = -((1 - 3/4)*div_J_prev + (3/4)*div_J_curr);

rho_mesh(:,:,end) = rho_mesh(:,:,end-1) + dt*(b1*rho_1 + b2*rho_2);