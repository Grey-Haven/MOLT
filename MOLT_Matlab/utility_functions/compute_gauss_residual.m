% ---------------------------------------------------------------------
% There are different ways to compute the residual in Gauss' Law
% 1. Compute E from E = -grad(phi) - dA/dt
%    Compute RES_field = div(E) - rho/sigma_1
% 2. From wave equation (d/dt)^2(phi) - Laplacian(phi) = rho/sigma_1
%    Compute RES_potential = (d/dt)^2(phi) - Laplacian(phi) -
%    rho/sigma_1.
% 3. From the Lorenz gauge, div(A) + 1/c^2 d/dt(phi) = 0
%    This implies d/dt(div(A)) + 1/c^2 (d/dt)^2(phi) = 0
%    From wave equation (d/dt)^2(phi) - Laplacian(phi) = rho/sigma_1
%    This implies -d/dt(div(A)) - Laplacian(phi) = rho/sigma_1
%    Compute RES_gauge = -d/dt(div(A)) - Laplacian(phi) - rho/sigma_1.
% ---------------------------------------------------------------------

% METHOD 1:
% Compute E = -grad(psi) - ddt_A
% For ddt A, we use backward finite-differences
% Note, E3 is not used in the particle update so we don't need ddt_A3
E1(:,:) = -ddx_psi(:,:,end) - ddt_A1(:,:);
E2(:,:) = -ddy_psi(:,:,end) - ddt_A2(:,:);

% Want to ensure the spatial derivatives we're taking are consistent.
if waves_update_method == waves_update_method_vanilla
    ddx_E1 = compute_ddx_FD(E1, dx);
    ddy_E2 = compute_ddy_FD(E2, dy);
elseif waves_update_method == waves_update_method_FFT
    ddx_E1 = compute_ddx_FFT(E1, kx_deriv_1);
    ddy_E2 = compute_ddy_FFT(E2, ky_deriv_1);
elseif waves_update_method == waves_update_method_FD6
    ME = MException('WaveException',"FD6 Derivative not implemented yet.");
    throw(ME);
    % ddx_E1 = compute_ddx_FD6(E1, dx);
    % ddy_E2 = compute_ddy_FD6(E2, dy);
elseif ismember(waves_update_method, [waves_update_method_poisson_phi, waves_update_method_pure_FFT])
    ddx_E1 = compute_ddx_FFT(E1, kx_deriv_1);
    ddy_E2 = compute_ddy_FFT(E2, ky_deriv_1);
else
    ME = MException('WaveException','Wave Method ' + wave_update_method + " not an option");
    throw(ME);
end

LHS_field = ddx_E1(:,:) + ddy_E2(:,:);

% METHOD 2:
ddt2_phi = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2))/(dt^2);

laplacian_phi_FFT = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);

LHS_potential = (1/(kappa^2))*ddt2_phi - laplacian_phi_FFT;

% METHOD 3:
div_A_curr = ddx_A1(:,:,end) + ddy_A2(:,:,end);
div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
ddt_div_A = (div_A_curr - div_A_prev)/dt;

LHS_gauge = -ddt_div_A - laplacian_phi_FFT;

% Compute all residuals
RHS = rho_mesh(:,:,end) / sigma_1;

gauss_law_potential_res = LHS_potential  - RHS;
gauss_law_gauge_res     = LHS_gauge      - RHS;
gauss_law_field_res     = LHS_field      - RHS;

% Store the residuals
gauss_law_potential_err_L2(steps+1) = get_L_2_error(gauss_law_potential_res, ...
                                                    zeros(size(gauss_residual(:,:))), ...
                                                    dx*dy);
gauss_law_gauge_err_L2(steps+1) = get_L_2_error(gauss_law_gauge_res, ...
                                                zeros(size(gauss_residual(:,:))), ...
                                                dx*dy);
gauss_law_field_err_L2(steps+1) = get_L_2_error(gauss_law_field_res, ...
                                                zeros(size(gauss_residual(:,:))), ...
                                                dx*dy);

gauss_law_potential_err_inf(steps+1) = max(max(abs(gauss_law_potential_res)));
gauss_law_gauge_err_inf(steps+1) = max(max(abs(gauss_law_gauge_res)));
gauss_law_field_err_inf(steps+1) = max(max(abs(gauss_law_field_res)));