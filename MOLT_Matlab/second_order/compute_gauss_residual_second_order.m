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

ddt_psi = (psi(:,:,end) - psi(:,:,end-1)) / dt;
ddt_A1  = (A1(:,:,end)  - A1(:,:,end-1))  / dt;
ddt_A2  = (A2(:,:,end)  - A2(:,:,end-1))  / dt;

E1(:,:) = -ddx_psi(:,:,end) - ddt_A1(:,:);
E2(:,:) = -ddy_psi(:,:,end) - ddt_A2(:,:);

% Want to ensure the spatial derivatives we're taking are consistent.
if ismember(waves_update_method, [waves_update_method_vanilla, waves_update_method_FD2])
    ddx_E1 = compute_ddx_FD(E1, dx);
    ddy_E2 = compute_ddy_FD(E2, dy);
elseif ismember(waves_update_method, [waves_update_method_FFT, waves_update_method_DIRK2])
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

div_E = ddx_E1(:,:) + ddy_E2(:,:);

% METHOD 2:
ddt2_phi = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2))/(dt^2);
% ddt2_phi = (ddt_psi_hist(:,:,end) - ddt_psi_hist(:,:,end-1)) / dt;

% avg_psi = (psi(:,:,end) + psi(:,:,end-2))/2;

b1 = 1/2;
b2 = 1/2;

c1 = 1/4;
c2 = 3/4;

% laplacian_avg_phi_FFT = compute_Laplacian_FFT(avg_psi,kx_deriv_2,ky_deriv_2);
laplacian_phi_FFT = compute_Laplacian_FFT(psi(:,:,end),kx_deriv_2,ky_deriv_2);

laplacian_phi_FFT_prev = compute_Laplacian_FFT(psi(:,:,end-1),kx_deriv_2,ky_deriv_2);
laplacian_phi_FFT_curr = compute_Laplacian_FFT(psi(:,:,end  ),kx_deriv_2,ky_deriv_2);

% laplacian_phi_1 = (1-c1)*laplacian_phi_FFT_prev + c1*laplacian_phi_FFT_curr;
% laplacian_phi_2 = (1-c2)*laplacian_phi_FFT_prev + c2*laplacian_phi_FFT_curr;

phi_1 = (1-c1) * psi(:,:,end-1) + c1*psi(:,:,end);
phi_2 = (1-c2) * psi(:,:,end-1) + c2*psi(:,:,end);

laplacian_phi_1 = compute_Laplacian_FFT(phi_1,kx_deriv_2,ky_deriv_2);
laplacian_phi_2 = compute_Laplacian_FFT(phi_2,kx_deriv_2,ky_deriv_2);

rho_prev = rho_mesh(:,:,end-1);
rho_curr = rho_mesh(:,:,end  );

rho_1 = (1-c1)*rho_prev + c1*rho_curr;
rho_2 = (1-c2)*rho_prev + c2*rho_curr;

rho_s = (b1*rho_1 + b2*rho_2) / sigma_1;
laplacian_phi_s = b1*laplacian_phi_1 + b2*laplacian_phi_2;

%%%%%%%%%%%%%%%%%%%%%%%
    a11 = 1/4;
    a12 = 0;
    a21 = 1/2;
    a22 = 1/4;

    u = psi(:,:,end-1);
    v = ( psi(:,:,end-1) - psi(:,:,end-2) ) / dt;
    h = dt;
    c = kappa;

    alpha1 = 1/(h*a11*c);
    alpha2 = 1/(h*a22*c);

    S_prev = rho_mesh(:,:,end-1) / sigma_1;
    S_curr = rho_mesh(:,:,end  ) / sigma_1;

    S_1 = (1-c1)*S_prev + c1*S_curr;
    S_2 = (1-c2)*S_prev + c2*S_curr;

    laplacian_u = compute_Laplacian_FFT(u, kx_deriv_2, ky_deriv_2);

    RHS1 = v + h*a11*c^2*(laplacian_u + S_1);
    u1 = solve_helmholtz_FFT(RHS1, alpha1, kx_deriv_2, ky_deriv_2);

    laplacian_u1 = compute_Laplacian_FFT(u1, kx_deriv_2, ky_deriv_2);

    v1 = c^2*(laplacian_u + S_1 + h*a11*laplacian_u1);

    RHS2 = v + h*a21*v1 + h*a22*c^2*(laplacian_u + h*a21*laplacian_u1 + S_2);

    u2 = solve_helmholtz_FFT(RHS2, alpha2, kx_deriv_2, ky_deriv_2);

    laplacian_u2 = compute_Laplacian_FFT(u2, kx_deriv_2, ky_deriv_2);

    v2 = c^2*(laplacian_u + h*a21*laplacian_u1 + h*a22*laplacian_u2 + S_2);

    u_next = u + h*(b1*u1 + b2*u2);
    v_next = v + h*(b1*v1 + b2*v2);

%%%%%%%%%%%%%%%%%%%%%%%

ddt_v = 1/kappa^2*(v_next - v)/dt;
u1 = (1-c1)*u + c1*u_next;
u2 = (1-c2)*u + c2*u_next;
v1 = (1-c1)*v + c1*v_next;
v2 = (1-c2)*v + c2*v_next;
laplacian_u1 = compute_Laplacian_FFT(u1,kx_deriv_2,ky_deriv_2);
laplacian_u2 = compute_Laplacian_FFT(u2,kx_deriv_2,ky_deriv_2);

% RHS = (b1*(rho_1 / sigma_1 + laplacian_phi_1) + b2*(rho_2 / sigma_1 + laplacian_phi_2));
RHS = rho_s + laplacian_phi_s;
% RHS = b1*rho_1/sigma_1 + b2*rho_2/sigma_1;

% LHS_potential = (1/(kappa^2))*ddt2_phi - laplacian_phi_FFT;

% METHOD 3:
div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
div_A_nm1  = ddx_A1(:,:,end-2) + ddy_A2(:,:,end-2);
% ddt_div_A = (div_A_curr - div_A_prev)/dt;

% ddx_ddt_A1 = compute_ddx_FFT(ddt_A1_hist(:,:,end), kx_deriv_1);
% ddy_ddt_A2 = compute_ddx_FFT(ddt_A2_hist(:,:,end), ky_deriv_1);
% 
% ddt_div_A = ddx_ddt_A1 + ddy_ddt_A2;

div_A_curr_1 = (1-c1)*div_A_prev + c1*div_A_curr;
div_A_curr_2 = (1-c2)*div_A_prev + c2*div_A_curr;

div_A_curr = b1*div_A_curr_1 + b2*div_A_curr_2;

div_A_prev_1 = (1-c1)*div_A_nm1  + c1*div_A_prev;
div_A_prev_2 = (1-c2)*div_A_nm1  + c2*div_A_prev;

div_A_prev = b1*div_A_prev_1 + b2*div_A_prev_2;

ddt_div_A = (div_A_curr - div_A_prev) / dt;

    % div_A_curr = ddx_A1(:,:,end  ) + ddy_A2(:,:,end  );
    % div_A_prev = ddx_A1(:,:,end-1) + ddy_A2(:,:,end-1);
    % div_A_nm1  = ddx_A1(:,:,end-2) + ddy_A2(:,:,end-2);
    % 
    % div_A_1 = (1-c1)*div_A_prev + c1*div_A_curr;
    % div_A_2 = (1-c2)*div_A_prev + c2*div_A_curr;
    % 
    % div_A_1_prev = (1-c1)*div_A_nm1 + c1*div_A_prev;
    % div_A_2_prev = (1-c2)*div_A_nm1 + c2*div_A_prev;
    % 
    % phi_1 = -div_A_1;
    % phi_2 = -div_A_2;
    % 
    % phi_1_prev = -div_A_1_prev;
    % phi_2_prev = -div_A_2_prev;
    % 
    % RHS_curr = b1*phi_1 + b2*phi_2;
    % RHS_prev = b1*phi_1_prev + b2*phi_2_prev;
    % 
    % ddt_div_A = (RHS_curr - RHS_prev) / dt;
    % ddt_gauge = (1/kappa^2)*ddt2_psi - ddt_div_A;

ddt_gauge = (1/kappa^2)*ddt2_psi + ddt_div_A;

LHS_gauge = -ddt_div_A - laplacian_phi_FFT;

% Compute all residuals
% RHS = (1 / sigma_1) * (rho_mesh(:,:,end) + rho_mesh(:,:,end-2)) / 2;
% RHS = rho_mesh(:,:,end) / sigma_1;

% gauss_law_potential_res = LHS_potential - RHS;
% gauss_law_gauge_res =     LHS_gauge - RHS;
% gauss_law_potential_res = (1/(kappa^2))*ddt2_phi  - RHS;
gauss_law_potential_res = (psi(:,:,end) - 2*psi(:,:,end-1) + psi(:,:,end-2)) - (kappa*dt)^2*RHS;
% gauss_law_gauge_res     = -ddt_div_A               - RHS;
gauss_law_gauge_res     = -(div_A_curr - div_A_prev) - dt*RHS;
% gauss_law_field_res     = div_E    - rho_curr / sigma_1;
gauss_law_field_res     = div_E                   - rho_s;

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