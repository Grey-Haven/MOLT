%---------------------------------------------------------------------
% 5.2.1. Advance the psi and its derivatives by dt using BDF-2 
%---------------------------------------------------------------------
psi(:,:,end) = BDF2_advance_dir(psi, psi_src, x, y, dx, dy, dt, kappa, beta_BDF2);
[div_psi_x, div_psi_y] = compute_gradient_FD4_dir(psi(:,:,end),dx,dy);
ddx_psi(:,:,end) = div_psi_x;
ddy_psi(:,:,end) = div_psi_y;

%---------------------------------------------------------------------
% 5.2.2. Advance the A1 and A2 and their derivatives by dt using BDF-2
%---------------------------------------------------------------------
A1(:,:,end) = BDF2_advance_dir(A1, A1_src, x, y, dx, dy, dt, kappa, beta_BDF2);
[div_A1_x, div_A1_y] = compute_gradient_FD4_dir(A1(:,:,end),dx,dy);
ddx_A1(:,:,end) = div_A1_x;
ddy_A1(:,:,end) = div_A1_y;

A2(:,:,end) = BDF2_advance_dir(A2, A2_src, x, y, dx, dy, dt, kappa, beta_BDF2);
[div_A2_x, div_A2_y] = compute_gradient_FD4_dir(A2(:,:,end),dx,dy);
ddx_A2(:,:,end) = div_A2_x;
ddy_A2(:,:,end) = div_A2_y;