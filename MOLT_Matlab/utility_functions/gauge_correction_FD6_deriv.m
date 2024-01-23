div_A = ddx_A1 + ddy_A2;

psi_prev = psi(:,:,end-1);
psi_A = psi(:,:,end);

psi_C = psi_prev - psi_A - kappa^2*dt*div_A;

psi(:,:,end) = psi_C + psi_A;

psi_next = psi(1:end-1,1:end-1,end);

[ddx_psi,ddy_psi] = compute_derivative_FD6_periodic(psi_next,dx,dy);