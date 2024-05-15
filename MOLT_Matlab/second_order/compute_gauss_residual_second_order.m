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

% A1_ave_np_half = (A1(:,:,end-0) + A1(:,:,end-1)) / 2;
% A1_ave_nm_half = (A1(:,:,end-1) + A1(:,:,end-2)) / 2;
% 
% A2_ave_np_half = (A2(:,:,end-0) + A2(:,:,end-1)) / 2;
% A2_ave_nm_half = (A2(:,:,end-1) + A2(:,:,end-2)) / 2;
% 
% ddx_A1_ave_np_half = (ddx_A1(:,:,end-0) + ddx_A1(:,:,end-1)) / 2;
% ddx_A1_ave_nm_half = (ddx_A1(:,:,end-1) + ddx_A1(:,:,end-2)) / 2;
% 
% ddy_A2_ave_np_half = (ddy_A2(:,:,end-0) + ddy_A2(:,:,end-1)) / 2;
% ddy_A2_ave_nm_half = (ddy_A2(:,:,end-1) + ddy_A2(:,:,end-2)) / 2;
% 
% ddt_A1 = (A1_ave_np_half - A1_ave_nm_half) / dt;
% ddt_A2 = (A2_ave_np_half - A2_ave_nm_half) / dt;

% ddt_psi = (psi(:,:,end) - psi(:,:,end-1)) / dt;
ddt_A1  = (A1(:,:,end)  - A1(:,:,end-1))  / dt;
ddt_A2  = (A2(:,:,end)  - A2(:,:,end-1))  / dt;

E1(:,:) = -ddx_psi(:,:,end) - ddt_A1(:,:);
E2(:,:) = -ddy_psi(:,:,end) - ddt_A2(:,:);
B3(:,:) = ddx_A2(:,:,end) - ddy_A1(:,:,end);

% Want to ensure the spatial derivatives we're taking are consistent.
if ismember(waves_update_method, [waves_update_method_vanilla, waves_update_method_FD2])
    ddx_E1 = compute_ddx_FD(E1, dx);
    ddy_E2 = compute_ddy_FD(E2, dy);
elseif ismember(waves_update_method, [waves_BDF_FFT_Family, waves_update_method_FFT, waves_update_method_DIRK2])
    ddx_E1 = compute_ddx_FFT(E1, kx_deriv_1);
    ddy_E2 = compute_ddy_FFT(E2, ky_deriv_1);
elseif waves_update_method == waves_update_method_FD4
    [ddx_E1,ddy_E1__] = compute_gradient_FD4_dir(E1,dx,dy);
    [ddx_E2__,ddy_E2] = compute_gradient_FD4_dir(E2,dx,dy);
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


% METHOD 1:
div_E = ddx_E1(:,:) + ddy_E2(:,:);

if waves_update_method == waves_update_method_DIRK2

    % b1 = 1/2;
    % b2 = 1/2;
    % 
    % c1 = 1/4;
    % c2 = 3/4;

    % Compute RHS:
    laplacian_phi_FFT = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);
    
    laplacian_phi_FFT_prev = compute_Laplacian_FFT(psi(:,:,end-1),kx_deriv_2,ky_deriv_2);
    laplacian_phi_FFT_curr = compute_Laplacian_FFT(psi(:,:,end  ),kx_deriv_2,ky_deriv_2);
    
    % phi_1 = (1-c1) * psi(:,:,end-1) + c1*psi(:,:,end);
    % phi_2 = (1-c2) * psi(:,:,end-1) + c2*psi(:,:,end);
    
    % laplacian_phi_1 = compute_Laplacian_FFT(phi_1,kx_deriv_2,ky_deriv_2);
    % laplacian_phi_2 = compute_Laplacian_FFT(phi_2,kx_deriv_2,ky_deriv_2);
    
    rho_prev = rho_mesh(:,:,end-1);
    rho_curr = rho_mesh(:,:,end  );
    
    % rho_1 = (1-c1)*rho_prev + c1*rho_curr;
    % rho_2 = (1-c2)*rho_prev + c2*rho_curr;
    
    % rho_s = (b1*rho_1 + b2*rho_2) / sigma_1;
    rho_s = DIRK2_d_RHS(rho_curr, rho_prev) / sigma_1;
    % laplacian_phi_s = b1*laplacian_phi_1 + b2*laplacian_phi_2;
    laplacian_phi_s = DIRK2_d_RHS(laplacian_phi_FFT_curr, laplacian_phi_FFT_prev);

    RHS = rho_s + laplacian_phi_s;
    % RHS = rho_curr + laplacian_phi_FFT_curr;

    div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
    div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
    div_A_nm1  = ddx_A1(:,:,end-2) + ddy_A2(:,:,end-2);
    
    % div_A_curr_1 = (1-c1)*div_A_prev + c1*div_A_curr;
    % div_A_curr_2 = (1-c2)*div_A_prev + c2*div_A_curr;
    
    % div_A_curr = b1*div_A_curr_1 + b2*div_A_curr_2;
    % div_A_curr = DIRK2_d_RHS(div_A_curr, div_A_prev);
    
    % div_A_prev_1 = (1-c1)*div_A_nm1  + c1*div_A_prev;
    % div_A_prev_2 = (1-c2)*div_A_nm1  + c2*div_A_prev;
    
    % div_A_prev = b1*div_A_prev_1 + b2*div_A_prev_2;
    % div_A_prev = DIRK2_d_RHS(div_A_prev, div_A_nm1);

    ddx_psi_hist = compute_ddx_FFT(ddt_psi_hist(:,:,end), kx_deriv_1);
    ddy_psi_hist = compute_ddy_FFT(ddt_psi_hist(:,:,end), ky_deriv_1);
    div_psi_hist = ddx_psi_hist + ddy_psi_hist;
    
    % Compute all residuals
    gauss_law_potential_res = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2)) - (kappa*dt)^2*RHS;
    % gauss_law_gauge_res     = -(div_A_curr - div_A_prev) - dt*RHS;
    gauss_law_gauge_res     = (div_A_curr - div_A_prev) + dt*RHS;
    gauss_law_field_res     = div_E                   - rho_s;

else
    % METHOD 2:
    if waves_update_method == waves_update_method_BDF1_FFT || waves_update_method == waves_update_method_pure_FFT || waves_update_method == waves_update_method_CDF1_FFT
        ddt2_phi = BDF1_d2(psi, dt);
    elseif waves_update_method == waves_update_method_BDF2_FFT
        ddt2_phi = BDF2_d2(psi, dt);
    elseif waves_update_method == waves_update_method_vanilla
        ddt2_phi = BDF2_d2(psi, dt);
    else
        ME = MException('WaveException','BDF Wave Method ' + wave_update_method + " not an option");
        throw(ME);
    end
    
    % METHOD 3:
    div_A_hist = ddx_A1 + ddy_A2;

    if waves_update_method == waves_update_method_BDF1_FFT || waves_update_method == waves_update_method_CDF1_FFT || waves_update_method == waves_update_method_pure_FFT
        ddt_div_A = BDF1_d(div_A_hist, dt);
    elseif waves_update_method == waves_update_method_BDF2_FFT
        ddt_div_A = BDF2_d(div_A_hist, dt);
    elseif waves_update_method == waves_update_method_vanilla
        ddt_div_A = BDF2_d(div_A_hist, dt);
    else
        ME = MException('WaveException','BDF Wave Method ' + wave_update_method + " not an option");
        throw(ME);
    end

    laplacian_phi_FFT = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);

    LHS_potential = (1/(kappa^2))*ddt2_phi - laplacian_phi_FFT;

    LHS_gauge = -ddt_div_A - laplacian_phi_FFT;

    RHS = rho_mesh(:,:,end) / sigma_1;

    % Compute all residuals
    gauss_law_potential_res = LHS_potential - RHS;
    gauss_law_gauge_res     = LHS_gauge     - RHS;
    gauss_law_field_res     = div_E         - RHS;

end

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