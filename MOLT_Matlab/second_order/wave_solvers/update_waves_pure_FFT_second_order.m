%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------
if waves_update_method == waves_update_method_BDF1_FFT

    alpha = beta_BDF1/(kappa*dt);
    psi_source_with_prev = BDF1_d2_update(psi(:,:,1:end-1), psi_src_hist(:,:,end), alpha);
    A1_source_with_prev  = BDF1_d2_update(A1(:,:,1:end-1), A1_src_hist(:,:,end), alpha);
    A2_source_with_prev  = BDF1_d2_update(A2(:,:,1:end-1), A2_src_hist(:,:,end), alpha);

elseif waves_update_method == waves_update_method_BDF2_FFT

    alpha = beta_BDF2/(kappa*dt);
    psi_source_with_prev = BDF2_d2_update(psi(:,:,1:end-1), psi_src_hist(:,:,end), alpha);
    A1_source_with_prev  = BDF2_d2_update(A1(:,:,1:end-1), A1_src_hist(:,:,end), alpha);
    A2_source_with_prev  = BDF2_d2_update(A2(:,:,1:end-1), A2_src_hist(:,:,end), alpha);

elseif waves_update_method == waves_update_method_BDF3_FFT

    alpha = beta_BDF3/(kappa*dt);
    psi_source_with_prev = BDF3_d2_update(psi(:,:,1:end-1), psi_src_hist(:,:,end), alpha);
    A1_source_with_prev  = BDF3_d2_update(A1(:,:,1:end-1), A1_src_hist(:,:,end), alpha);
    A2_source_with_prev  = BDF3_d2_update(A2(:,:,1:end-1), A2_src_hist(:,:,end), alpha);

elseif waves_update_method == waves_update_method_CDF1_FFT
    a = 1/(kappa*dt);
    alpha = beta_CDF1/a;
    laplacian_psi_prev = compute_Laplacian_FFT(psi(:,:,end-2), kx_deriv_2, ky_deriv_2);
    laplacian_A1_prev  = compute_Laplacian_FFT(A1(:,:,end-2) , kx_deriv_2, ky_deriv_2);
    laplacian_A2_prev  = compute_Laplacian_FFT(A2(:,:,end-2) , kx_deriv_2, ky_deriv_2);

    psi_source_with_prev = 1/a^2 * ((rho_mesh(:,:,end) + rho_mesh(:,:,end-1))/(2*sigma_1) + 1/2*laplacian_psi_prev + 2*psi(:,:,end-1) - psi(:,:,end-2));
    A1_source_with_prev  = 1/a^2 * (sigma_2*(J1_mesh(:,:,end) + J1_mesh(:,:,end-1))/2 + 1/2*laplacian_A1_prev + 2*A1(:,:,end-1) - A1(:,:,end-2));
    A2_source_with_prev  = 1/a^2 * (sigma_2*(J2_mesh(:,:,end) + J2_mesh(:,:,end-1))/2 + 1/2*laplacian_A2_prev + 2*A2(:,:,end-1) - A2(:,:,end-2));
else
    ME = MException('WaveException','BDF Wave Method ' + wave_update_method + " not an option");
    throw(ME);
end

psi(:,:,end) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2);
A1(:,:,end)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);
A2(:,:,end)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);