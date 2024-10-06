div_A = ddx_A1(:,:,end) + ddy_A2(:,:,end);

psi_prev = psi(:,:,end-1);
psi_A = psi(:,:,end);

psi_C = psi_prev - psi_A - kappa^2*dt*div_A;

psi(:,:,end) = psi_C + psi_A;

psi_next = psi(1:end-1,1:end-1,end);

[ddx_psi,ddy_psi] = compute_gradient_FD6_per(psi_next,dx,dy);