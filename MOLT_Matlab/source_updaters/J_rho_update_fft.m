% Compute the next step of rho using the continuity equation.
% The FFT will be used to compute div(J).

J_compute_vanilla;

J1_clean(1:end-1,1:end-1) = ifft(fft(ifft(fft(J1_mesh,N_x-1,1),N_x-1,1),N_y-1,2),N_y-1,2);
J2_clean(1:end-1,1:end-1) = ifft(fft(ifft(fft(J2_mesh,N_x-1,1),N_x-1,1),N_y-1,2),N_y-1,2);
J1_clean(end,:) = J1_clean(1,:);
J1_clean(:,end) = J1_clean(:,1);
J2_clean(end,:) = J2_clean(1,:);
J2_clean(:,end) = J2_clean(:,1);

J1_clean_FFTx = fft(J1_clean(1:end-1,1:end-1),N_x-1,1);
J2_clean_FFTy = fft(J2_clean(1:end-1,1:end-1),N_y-1,2);

J1_deriv_clean(1:end-1,1:end-1) = ifft(sqrt(-1)*kx'.*J1_clean_FFTx,N_x-1,1);
J2_deriv_clean(1:end-1,1:end-1) = ifft(sqrt(-1)*ky .*J2_clean_FFTy,N_y-1,2);
J1_deriv_clean(end,:) = J1_deriv_clean(1,:);
J1_deriv_clean(:,end) = J1_deriv_clean(:,1);
J2_deriv_clean(end,:) = J2_deriv_clean(1,:);
J2_deriv_clean(:,end) = J2_deriv_clean(:,1);

Gamma = -1/((N_x-1)*(N_y-1))*sum(sum(J1_deriv_clean(1:end-1,1:end-1) + J2_deriv_clean(1:end-1,1:end-1)));

F1 = .5*Gamma*x'.*ones(N_y,N_x);
F2 = .5*Gamma*y .*ones(N_y,N_x);

J1_star = J1_clean + F1;
J2_star = J1_clean + F2;

J1_star_FFTx = fft(J1_star(1:end-1,1:end-1),N_x-1,1);
J2_star_FFTy = fft(J2_star(1:end-1,1:end-1),N_y-1,2);

J1_star_deriv = ifft(sqrt(-1)*kx'.*J1_star_FFTx,N_x-1,1);
J2_star_deriv = ifft(sqrt(-1)*ky .*J2_star_FFTy,N_y-1,2);

J1_deriv_clean(1:end-1,1:end-1) = J1_star_deriv;
J2_deriv_clean(1:end-1,1:end-1) = J2_star_deriv;
J1_deriv_clean(end,:) = J1_deriv_clean(1,:);
J1_deriv_clean(:,end) = J1_deriv_clean(:,1);
J2_deriv_clean(end,:) = J2_deriv_clean(1,:);
J2_deriv_clean(:,end) = J2_deriv_clean(:,1);

rho_mesh = rho_mesh - dt*(J1_deriv_clean + J2_deriv_clean);

J1_mesh = J1_star;
J2_mesh = J2_star;

J_mesh(:,:,1) = J1_mesh;
J_mesh(:,:,2) = J2_mesh;