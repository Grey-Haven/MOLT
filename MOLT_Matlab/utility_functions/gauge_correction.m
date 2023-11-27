div_A = ddx_A1 + ddy_A2;

psi_prev = psi(:,:,end-1);
psi_A = psi(:,:,end);

psi_C = psi_prev - psi_A - kappa^2*dt*div_A;

psi(:,:,end) = psi_C + psi_A;