%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------


if waves_update_method == waves_update_method_BDF1_FFT

    alpha = beta_BDF1/(kappa*dt);
    psi_source_with_prev = BDF1_d2_update(psi(:,:,1:end-1), psi_src, alpha);
    A1_source_with_prev  = BDF1_d2_update(A1(:,:,1:end-1), A1_src, alpha);
    A2_source_with_prev  = BDF1_d2_update(A2(:,:,1:end-1), A2_src, alpha);

elseif waves_update_method == waves_update_method_BDF2_FFT

    alpha = beta_BDF2/(kappa*dt);
    psi_source_with_prev = BDF2_d2_update(psi(:,:,1:end-1), psi_src, alpha);
    A1_source_with_prev  = BDF2_d2_update(A1(:,:,1:end-1), A1_src, alpha);
    A2_source_with_prev  = BDF2_d2_update(A2(:,:,1:end-1), A2_src, alpha);

elseif waves_update_method == waves_update_method_BDF3_FFT

    alpha = beta_BDF3/(kappa*dt);
    psi_source_with_prev = BDF3_d2_update(psi(:,:,1:end-1), psi_src, alpha);
    A1_source_with_prev  = BDF3_d2_update(A1(:,:,1:end-1), A1_src, alpha);
    A2_source_with_prev  = BDF3_d2_update(A2(:,:,1:end-1), A2_src, alpha);

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