dt_full = -dt;
dt_half = -dt / 2;
dt = dt_half;

%---------------------------------------------------------------------
% 1. Advance electron positions by dt using v^{n}
%---------------------------------------------------------------------

v1_star = 2*v1_elec_old - v1_elec_new;
v2_star = 2*v2_elec_old - v2_elec_new;
[x1_elec_nm1, x2_elec_nm1] = advance_particle_positions_2D(x1_elec_old, x2_elec_old, ...
                                                           v1_star, v2_star, dt_full);


% Apply the particle boundary conditions
% Need to include the shift function here
x1_elec_nm1 = periodic_shift(x1_elec_nm1, x(1), L_x);
x2_elec_nm1 = periodic_shift(x2_elec_nm1, y(1), L_y);

%---------------------------------------------------------------------
% 2. Compute the electron current density used for updating A
%    Compute also the charge density used for updating psi
%---------------------------------------------------------------------
if J_rho_update_method == J_rho_update_method_vanilla
    J_rho_update_vanilla;
elseif J_rho_update_method == J_rho_update_method_FFT
    % Updates J (half time) based on averaged location (integer time) and 
    % velocity (half time) then updates rho based on
    % continuity equation (integer time)
    x1_elec_ave = (x1_elec_nm1 + x1_elec_old)/2;
    x2_elec_ave = (x2_elec_nm1 + x2_elec_old)/2;
    
    J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                                x1_elec_ave, x2_elec_ave, ...
                                v1_elec_old, v2_elec_old, ...
                                q_elec, cell_volumes, w_elec);
    
    % Need to enforce periodicity for the current on the mesh
    J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
    J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));
    
    J1_mesh(:,:,end-1) = J_mesh(:,:,1);
    J2_mesh(:,:,end-1) = J_mesh(:,:,2);

    ddx_J1 = compute_ddx_FFT(J_mesh(:,:,1), kx_deriv_1);
    ddy_J2 = compute_ddy_FFT(J_mesh(:,:,2), ky_deriv_1);

    rho_mesh(:,:,end-2) = rho_mesh(:,:,end-1) - dt_full * (ddx_J1 + ddy_J2);
elseif J_rho_update_method == J_rho_update_method_FD6
    J_rho_update_FD6;
else
    ME = MException('SourceException','Source Method ' + J_rho_update_method + " not an option");
    throw(ME);
end

%---------------------------------------------------------------------
% 3.1. Compute wave sources
%---------------------------------------------------------------------
psi_src(:,:) = (1/sigma_1)*(rho_mesh(:,:,end) + rho_mesh(:,:,end-2));
A1_src(:,:) = sigma_2*(J1_mesh(:,:,end-1) + J1_mesh_p3_2);
A2_src(:,:) = sigma_2*(J2_mesh(:,:,end-1) + J2_mesh_p3_2);

%---------------------------------------------------------------------
% 3.2 Update the scalar (phi) and vector (A) potentials waves. 
%---------------------------------------------------------------------
if waves_update_method == waves_update_method_vanilla
    update_waves_vanilla_second_order;
elseif waves_update_method == waves_update_method_FFT
    update_waves_hybrid_FFT_second_order;
elseif waves_update_method == waves_update_method_FD6
    update_waves_hybrid_FD6_second_order;
elseif waves_update_method == waves_update_method_poisson_phi
    update_waves_poisson_phi_second_order;
elseif waves_update_method == waves_update_method_pure_FFT
%     update_waves_pure_FFT_second_order;
    alpha_half_step = beta_CDF2/(kappa*dt_half);
    alpha_full_step = beta_CDF2/(kappa*dt_full);
    
    psi_source_with_prev = 2*psi(:,:,end-1) + 1/(alpha_full_step^2)*psi_src;
    A1_source_with_prev  = 2*A1(:,:,end-1)  + 1/(alpha_half_step^2)*A1_src;
    A2_source_with_prev  = 2*A2(:,:,end-1)  + 1/(alpha_half_step^2)*A2_src;
    %---------------------------------------------------------------------
    % 5.2.1. Advance the waves (replacing A_0 with A_{-1/2}
    %---------------------------------------------------------------------
    A1(:,:,end-1)  = solve_helmholtz_FFT(A1_source_with_prev , alpha_half_step, kx_deriv_2, ky_deriv_2) - A1_mesh_p3_2;
    A2(:,:,end-1)  = solve_helmholtz_FFT(A2_source_with_prev , alpha_half_step, kx_deriv_2, ky_deriv_2) - A2_mesh_p3_2;
    psi(:,:,end-2) = solve_helmholtz_FFT(psi_source_with_prev, alpha_full_step, kx_deriv_2, ky_deriv_2) - psi(:,:,end);
    
    %---------------------------------------------------------------------
    % 5.2.1. Compute their derivatives using the FFT
    %---------------------------------------------------------------------    
    ddx_A1(:,:,end-1)   = compute_ddx_FFT(A1(:,:,end-2) , kx_deriv_1);
    ddy_A1(:,:,end-1)   = compute_ddy_FFT(A1(:,:,end-2) , ky_deriv_1);
    
    ddx_A2(:,:,end-1)   = compute_ddx_FFT(A2(:,:,end-2) , kx_deriv_1);
    ddy_A2(:,:,end-1)   = compute_ddy_FFT(A2(:,:,end-2) , ky_deriv_1);
    
    ddx_psi(:,:,end-2)  = compute_ddx_FFT(psi(:,:,end-2), kx_deriv_1);
    ddy_psi(:,:,end-2)  = compute_ddy_FFT(psi(:,:,end-2), ky_deriv_1);
else
    ME = MException('WaveException','Wave Method ' + wave_update_method + " not an option");
    throw(ME);
end

%---------------------------------------------------------------------
% 3.3 Correct gauge error (optional)
%---------------------------------------------------------------------
if gauge_correction == gauge_correction_none
    % Nothing
elseif gauge_correction == gauge_correction_FFT
    gauge_correction_FFT_deriv;
elseif gauge_correction == gauge_correction_FD6
    gauge_correction_FD6_deriv;
else
    ME = MException('GaugeCorrectionException','Gauge Correction Method ' + gauge_correction + " not an option");
    throw(ME);
end


%---------------------------------------------------------------------
% 4. Momentum advance by dt
%---------------------------------------------------------------------
ddx_psi_ave = (ddx_psi(:,:,end-1) + ddx_psi(:,:,end-2)) / 2;
ddy_psi_ave = (ddy_psi(:,:,end-1) + ddy_psi(:,:,end-2)) / 2;

[v1_elec_nm1, v2_elec_nm1, P1_elec_nm1, P2_elec_nm1] = ...
improved_asym_euler_momentum_push_2D2P_implicit(x1_elec_nm1, x2_elec_nm1, ...
                                                P1_elec_old, P2_elec_old, ...
                                                v1_elec_old, v2_elec_old, ...
                                                v1_elec_new, v2_elec_new, ...
                                                ddx_psi_ave, ddy_psi_ave, ...
                                                A1(:,:,end-2), ddx_A1(:,:,end-2), ddy_A1(:,:,end-2), ...
                                                A2(:,:,end-2), ddx_A2(:,:,end-2), ddy_A2(:,:,end-2), ...
                                                x, y, dx, dy, q_elec, r_elec, ...
                                                kappa, dt_half);