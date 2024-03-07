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

% J1_deriv_clean(1:end-1,1:end-1) = J1_star_deriv;
% J2_deriv_clean(1:end-1,1:end-1) = J2_star_deriv;

if J_rho_update_method == J_rho_update_method_BDF1_FFT
    rho_mesh(1:end-1,1:end-1,end) = rho_mesh(1:end-1,1:end-1,end-1) - dt*(J1_star_deriv + J2_star_deriv);
elseif J_rho_update_method == J_rho_update_method_BDF2_FFT
    rho_mesh(1:end-1,1:end-1,end) = 4/3*rho_mesh(1:end-1,1:end-1,end-1) - 1/3*rho_mesh(1:end-1,1:end-1,end-2) - ((2/3)*dt)*(J1_star_deriv + J2_star_deriv);
elseif J_rho_update_method == J_rho_update_method_BDF3_FFT
    rho_mesh(1:end-1,1:end-1,end) = 18/11*rho_mesh(1:end-1,1:end-1,end-1) - 9/11*rho_mesh(1:end-1,1:end-1,end-2) + 2/11*rho_mesh(1:end-1,1:end-1,end-3) - ((6/11)*dt)*(J1_star_deriv + J2_star_deriv);
elseif J_rho_update_method == J_rho_update_method_BDF4_FFT
    rho_mesh(1:end-1,1:end-1,end) = 48/25*rho_mesh(1:end-1,1:end-1,end-1) - 36/25*rho_mesh(1:end-1,1:end-1,end-2) + 16/25*rho_mesh(1:end-1,1:end-1,end-3) - 3/25*rho_mesh(1:end-1,1:end-1,end-4) - ((12/25)*dt)*(J1_star_deriv + J2_star_deriv);
else
    ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
    throw(ME);
end
% rho_mesh(1:end-1,1:end-1,end) = rho_mesh(1:end-1,1:end-1,end-1) - dt*(J1_star_deriv + J2_star_deriv);
% rho_mesh(1:end-1,1:end-1,end) = 4/3*rho_mesh(1:end-1,1:end-1,end-1) - 1/3*rho_mesh(1:end-1,1:end-1,end-2) - ((2/3)*dt)*(J1_star_deriv + J2_star_deriv);
% rho_mesh(1:end-1,1:end-1,end) = 18/11*rho_mesh(1:end-1,1:end-1,end-1) - 9/11*rho_mesh(1:end-1,1:end-1,end-2) + 2/11*rho_mesh(1:end-1,1:end-1,end-3) - ((6/11)*dt)*(J1_star_deriv + J2_star_deriv);
% rho_mesh(1:end-1,1:end-1,end) = 48/25*rho_mesh(1:end-1,1:end-1,end-1) - 36/25*rho_mesh(1:end-1,1:end-1,end-2) + 16/25*rho_mesh(1:end-1,1:end-1,end-3) - 3/25*rho_mesh(1:end-1,1:end-1,end-4) - ((12/25)*dt)*(J1_star_deriv + J2_star_deriv);

rho_mesh(:,:,end) = copy_periodic_boundaries(rho_mesh(:,:,end));

J1_mesh(1:end-1,1:end-1,end) = J1_star;
J2_mesh(1:end-1,1:end-1,end) = J2_star;

J1_mesh(:,:,end) = copy_periodic_boundaries(J1_mesh(:,:,end));
J2_mesh(:,:,end) = copy_periodic_boundaries(J2_mesh(:,:,end));