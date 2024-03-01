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

J1_star_FFTx = fft(J1_star,N_x-1,2);
J2_star_FFTy = fft(J2_star,N_y-1,1);

J1_star_deriv = ifft(sqrt(-1)*kx_deriv_1 .*J1_star_FFTx,N_x-1,2);
J2_star_deriv = ifft(sqrt(-1)*ky_deriv_1'.*J2_star_FFTy,N_y-1,1);

J1_mesh(1:end-1,1:end-1,end) = J1_star;
J2_mesh(1:end-1,1:end-1,end) = J2_star;

J1_mesh(:,:,end) = copy_periodic_boundaries(J1_mesh(:,:,end));
J2_mesh(:,:,end) = copy_periodic_boundaries(J2_mesh(:,:,end));

ddx_J1 = compute_ddx_FFT(J1_mesh(:,:,end-1), kx_deriv_1);
ddy_J2 = compute_ddy_FFT(J2_mesh(:,:,end-1), ky_deriv_1);
div_J_prev = ddx_J1 + ddy_J2;

ddx_J1 = compute_ddx_FFT(J1_mesh(:,:,end  ), kx_deriv_1);
ddy_J2 = compute_ddy_FFT(J2_mesh(:,:,end  ), ky_deriv_1);
div_J_curr = ddx_J1 + ddy_J2;

b1 = 1/2;
b2 = 1/2;

c1 = 1/4;
c2 = 3/4;

% J1_prev = J1_mesh(:,:,end-1);
% J2_prev = J2_mesh(:,:,end-1);
% 
% J1_curr = J1_mesh(:,:,end-0);
% J2_curr = J2_mesh(:,:,end-0);
% 
% J1_1 = (1-c1)*J1_prev + c1*J1_curr;
% J1_2 = (1-c2)*J1_prev + c2*J1_curr;
% 
% J2_1 = (1-c1)*J2_prev + c1*J2_curr;
% J2_2 = (1-c2)*J2_prev + c2*J2_curr;
% 
% ddx_J1_1 = compute_ddx_FFT(J1_1, kx_deriv_1);
% ddx_J1_2 = compute_ddx_FFT(J1_2, kx_deriv_1);
% 
% ddy_J2_1 = compute_ddy_FFT(J2_1, ky_deriv_1);
% ddy_J2_2 = compute_ddy_FFT(J2_2, ky_deriv_1);
% 
% div_J_1 = ddx_J1_1 + ddy_J2_1;
% div_J_2 = ddx_J1_2 + ddy_J2_2;
% 
% rho_1 = -div_J_1;
% rho_2 = -div_J_2;

div_J_1 = (1-c1)*div_J_prev + c1*div_J_curr;
div_J_2 = (1-c2)*div_J_prev + c2*div_J_curr;

rho_1 = -div_J_1;
rho_2 = -div_J_2;

rho_mesh(:,:,end) = rho_mesh(:,:,end-1) + dt*(b1*rho_1 + b2*rho_2);