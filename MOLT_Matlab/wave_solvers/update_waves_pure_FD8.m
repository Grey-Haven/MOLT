%---------------------------------------------------------------------
% 5.2.1. Advance the waves
%---------------------------------------------------------------------

if waves_update_method == waves_update_method_BDF1_FD8

    alpha = beta_BDF1/(kappa*dt);
    psi_source_with_prev = BDF1_d2_update(psi(:,:,1:end-1), (1/sigma_1)*rho_mesh(:,:,end), alpha);
    A1_source_with_prev  = BDF1_d2_update(A1(:,:,1:end-1), sigma_2*J1_mesh(:,:,end), alpha);
    A2_source_with_prev  = BDF1_d2_update(A2(:,:,1:end-1), sigma_2*J2_mesh(:,:,end), alpha);

elseif waves_update_method == waves_update_method_BDF2_FD8

    alpha = beta_BDF2/(kappa*dt);
    psi_source_with_prev = BDF2_d2_update(psi(:,:,1:end-1), (1/sigma_1)*rho_mesh(:,:,end), alpha);
    A1_source_with_prev  = BDF2_d2_update(A1(:,:,1:end-1), sigma_2*J1_mesh(:,:,end), alpha);
    A2_source_with_prev  = BDF2_d2_update(A2(:,:,1:end-1), sigma_2*J2_mesh(:,:,end), alpha);

else
    ME = MException('WaveException','BDF Wave Method ' + wave_update_method + " not an option");
    throw(ME);
end

guess = zeros(size(psi(1:end-1,1:end-1,end)));

psi(1:end-1,1:end-1,end) = solve_helmholtz_MG_FD8(guess, psi_source_with_prev(1:end-1,1:end-1), alpha, dx, dy);

A1(1:end-1,1:end-1,end)  = solve_helmholtz_MG_FD8(guess, A1_source_with_prev(1:end-1,1:end-1) , alpha, dx, dy);

A2(1:end-1,1:end-1,end)  = solve_helmholtz_MG_FD8(guess, A2_source_with_prev(1:end-1,1:end-1) , alpha, dx, dy);

psi(:,:,end) = copy_periodic_boundaries(psi(:,:,end));
A1(:,:,end) = copy_periodic_boundaries(A1(:,:,end));
A2(:,:,end) = copy_periodic_boundaries(A2(:,:,end));

%---------------------------------------------------------------------
% 5.2.1. Compute their derivatives using the FD8
%---------------------------------------------------------------------

[ddx_psi(:,:,end), ddy_psi(:,:,end)] = compute_gradient_FD8_per(psi(1:end-1,1:end-1,end), dx, dy);

[ddx_A1(:,:,end) , ddy_A1(:,:,end) ] = compute_gradient_FD8_per(A1(1:end-1,1:end-1,end) , dx, dy);

[ddx_A2(:,:,end) , ddy_A2(:,:,end) ] = compute_gradient_FD8_per(A2(1:end-1,1:end-1,end) , dx, dy);