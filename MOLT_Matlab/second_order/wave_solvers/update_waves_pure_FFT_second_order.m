%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------
if waves_update_method == waves_update_method_BDF1_FFT

    alpha = beta_BDF1/(kappa*dt);
    psi_source_with_prev = BDF1_d2_update(psi(:,:,1:end-1), (1/sigma_1)*rho_mesh(:,:,end), alpha);
    A1_source_with_prev  = BDF1_d2_update(A1(:,:,1:end-1), sigma_2*J1_mesh(:,:,end), alpha);
    A2_source_with_prev  = BDF1_d2_update(A2(:,:,1:end-1), sigma_2*J2_mesh(:,:,end), alpha);

elseif waves_update_method == waves_update_method_BDF2_FFT

    alpha = beta_BDF2/(kappa*dt);
    psi_source_with_prev = BDF2_d2_update(psi(:,:,1:end-1), (1/sigma_1)*rho_mesh(:,:,end), alpha);
    A1_source_with_prev  = BDF2_d2_update(A1(:,:,1:end-1), sigma_2*J1_mesh(:,:,end), alpha);
    A2_source_with_prev  = BDF2_d2_update(A2(:,:,1:end-1), sigma_2*J2_mesh(:,:,end), alpha);

elseif waves_update_method == waves_update_method_BDF3_FFT

    alpha = beta_BDF3/(kappa*dt);
    psi_source_with_prev = BDF3_d2_update(psi(:,:,1:end-1), (1/sigma_1)*rho_mesh(:,:,end), alpha);
    A1_source_with_prev  = BDF3_d2_update(A1(:,:,1:end-1), sigma_2*J1_mesh(:,:,end), alpha);
    A2_source_with_prev  = BDF3_d2_update(A2(:,:,1:end-1), sigma_2*J2_mesh(:,:,end), alpha);

elseif waves_update_method == waves_update_method_CDF2_FFT

    % The averaging term 1/2 in eg (rho^n+1 + rho^n-1)/2 is in the alpha
    % term (beta_CDF1 = sqrt(2))

    alpha = beta_CDF2/(kappa*dt);
    psi_source_with_prev = 1/alpha^2*((rho_mesh(:,:,end) + rho_mesh(:,:,end-2))/sigma_1) + 2*psi(:,:,end-1);
    A1_source_with_prev  = sigma_2/alpha^2*(J1_mesh(:,:,end) + J1_mesh(:,:,end-2)) + 2*A1(:,:,end-1);
    A2_source_with_prev  = sigma_2/alpha^2*(J2_mesh(:,:,end) + J2_mesh(:,:,end-2)) + 2*A2(:,:,end-1);
else
    ME = MException('WaveException','BDF Wave Method ' + wave_update_method + " not an option");
    throw(ME);
end

if waves_update_method == waves_update_method_CDF2_FFT
    psi(:,:,end) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2) - psi(:,:,end-2);
    A1(:,:,end)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A1(:,:,end-2);
    A2(:,:,end)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2) - A2(:,:,end-2);
else
    psi(:,:,end) = solve_helmholtz_FFT(psi_source_with_prev, alpha, kx_deriv_2, ky_deriv_2);
    A1(:,:,end)  = solve_helmholtz_FFT(A1_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);
    A2(:,:,end)  = solve_helmholtz_FFT(A2_source_with_prev , alpha, kx_deriv_2, ky_deriv_2);
end

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FFT
%---------------------------------------------------------------------

ddx_psi(:,:,end) = compute_ddx_FFT(psi(:,:,end), kx_deriv_1);
ddy_psi(:,:,end) = compute_ddy_FFT(psi(:,:,end), ky_deriv_1);

ddx_A1(:,:,end)  = compute_ddx_FFT(A1(:,:,end) , kx_deriv_1);
ddy_A1(:,:,end)  = compute_ddy_FFT(A1(:,:,end) , ky_deriv_1);

ddx_A2(:,:,end)  = compute_ddx_FFT(A2(:,:,end) , kx_deriv_1);
ddy_A2(:,:,end)  = compute_ddy_FFT(A2(:,:,end) , ky_deriv_1);