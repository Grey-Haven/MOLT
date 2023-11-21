% Compute the next step of rho using the continuity equation.
% We get J^n from rho^n*u_avg

rho_compute_vanilla;

u_avg_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_elec_new, x2_elec_new, v1_elec_old, v2_elec_old, v1_elec_new, v2_elec_new);
n_dens  = scatter_2D_vectorized(N_x, N_y, x1_elec_new, x2_elec_new, x', y', dx, dy, 1);
u_avg_mesh(:,:,1) = enforce_periodicity(u_avg_mesh(:,:,1));
u_avg_mesh(:,:,2) = enforce_periodicity(u_avg_mesh(:,:,2));
n_dens(:,:) = enforce_periodicity(n_dens(:,:));

u_avg_mesh(:,:,1) = u_avg_mesh(:,:,1) ./ n_dens;
u_avg_mesh(:,:,2) = u_avg_mesh(:,:,2) ./ n_dens;
u_avg_mesh(isnan(u_avg_mesh)) = 0;

u_avg_mesh(1:end-1,1:end-1,1) = ifft(ifft(fft(fft(u_avg_mesh(1:end-1,1:end-1,1),N_x-1,1),N_y-1,2),N_x-1,1),N_y-1,2);
u_avg_mesh(end,:,1) = u_avg_mesh(1,:,1);
u_avg_mesh(:,end,1) = u_avg_mesh(:,1,1);

u_avg_mesh(1:end-1,1:end-1,2) = ifft(ifft(fft(fft(u_avg_mesh(1:end-1,1:end-1,2),N_x-1,1),N_y-1,2),N_x-1,1),N_y-1,2);
u_avg_mesh(end,:,2) = u_avg_mesh(1,:,2);
u_avg_mesh(:,end,2) = u_avg_mesh(:,1,2);
        
rho_n = rho_elec(:,:);
opts = optimoptions('fsolve', 'TolFun', 1E-10, 'TolX', 1E-10, 'Display', 'Off');
rho_root = @(rho_guess) rho_implicit(rho_guess,rho_n(1:end-1,1:end-1),u_avg_mesh(1:end-1,1:end-1,:),dt,kx,ky);

rho_elec_next = fsolve(rho_root,rho_elec(1:end-1,1:end-1),opts);

sum_rho_curr = sum(sum(rho_elec(1:end-1,1:end-1)));
sum_rho_next = sum(sum(rho_elec_next));

Gamma = sum_rho_curr / sum_rho_next;

rho_elec_next = Gamma * rho_elec_next;

% Within the iterative solve:
J1_mesh(1:end-1,1:end-1) = rho_elec_next.*u_avg_mesh(1:end-1,1:end-1,1);
J2_mesh(1:end-1,1:end-1) = rho_elec_next.*u_avg_mesh(1:end-1,1:end-1,2);

J1_mesh(end,:) = J1_mesh(1,:);
J1_mesh(:,end) = J1_mesh(:,1);
J2_mesh(end,:) = J2_mesh(1,:);
J2_mesh(:,end) = J2_mesh(:,1);

% J1_fft_deriv_x = ifft(sqrt(-1)*kx'.*fft(J1_mesh(1:end-1,1:end-1),N_x-1,1),N_x-1,1);
% J2_fft_deriv_y = ifft(sqrt(-1)*ky .*fft(J2_mesh(1:end-1,1:end-1),N_y-1,2),N_y-1,2);
% 
% div_J = zeros(N_x,N_y);
% div_J(1:end-1,1:end-1) = J1_fft_deriv_x + J2_fft_deriv_y;
% div_J(end,:) = div_J(1,:);
% div_J(:,end) = div_J(:,1);

% disp(norm(norm(rho_elec_next - (rho_n - dt*div_J))));
% Asserting that the continuity equation is satisfied
% assert(norm(norm(rho_elec_next - (rho_n - dt*div_J))) < 10*eps);

rho_elec(1:end-1,1:end-1) = rho_elec_next;
rho_elec(end,:) = rho_elec(1,:);
rho_elec(:,end) = rho_elec(:,1);

rho_mesh = rho_ions + rho_elec;

J_mesh(:,:,1) = J1_mesh;
J_mesh(:,:,2) = J2_mesh;