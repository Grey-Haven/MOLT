%---------------------------------------------------------------------
% 5.2.1. Advance the psi and its derivatives by dt using BDF-1 
%---------------------------------------------------------------------

% Charge density is at the new time level from step (3)
% which is consistent with the BDF scheme
[psi, ddx_psi, ddy_psi] = BDF1_combined_per_advance_hybrid_FD6(psi, ddx_psi, ddy_psi, psi_src(:,:), ...
                                                               x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);

%---------------------------------------------------------------------
% 5.2.2. Advance the A1 and A2 and their derivatives by dt using BDF-1
%---------------------------------------------------------------------



% A1 uses J1
[A1, ddx_A1, ddy_A1] = BDF1_combined_per_advance_hybrid_FD6(A1, ddx_A1, ddy_A1, A1_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);

% A2 uses J2
[A2, ddx_A2, ddy_A2] = BDF1_combined_per_advance_hybrid_FD6(A2, ddx_A2, ddy_A2, A2_src(:,:), ...
                                                            x, y, t_n, dx, dy, dt, kappa, beta_BDF, kx_deriv_1, ky_deriv_1);