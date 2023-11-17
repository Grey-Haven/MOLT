% Compute splitting error with FFT
% gamma = (1/alpha^4)(dxx)(dyy){A1,A2}

alpha = beta_BDF/(kappa*dt);

gamma1 = zeros(size(A1(:,:,3)));
gamma2 = zeros(size(A2(:,:,3)));

A1_curr = A1(1:end-1,1:end-1,3);
A2_curr = A2(1:end-1,1:end-1,3);

A1_xx = ifft(-kx'.^2.*fft(A1_curr,N_x-1,1),N_x-1);
gamma1(1:end-1,1:end-1) = ifft(-ky.^2.*fft(A1_xx,N_y-1,2),N_y-1,2);
gamma1(:,end) = gamma1(:,1);
gamma1(end,:) = gamma1(1,:);


ERR1 = get_L_x_inverse_per(gamma1, x, y, dx, dy, dt, kappa, beta_BDF);
ERR1 = (1/alpha^4)*get_L_y_inverse_per(ERR1, x, y, dx, dy, dt, kappa, beta_BDF);

A1(:,:,3) = A1(:,:,3) - ERR1;
% A1(:,end) = A1(:,1);
% A1(end,:) = A1(1,:);

A2_xx = ifft(-kx'.^2.*fft(A2_curr(1:end-1,1:end-1),N_x-1,1),N_x-1);
gamma2(1:end-1,1:end-1) = ifft(-ky.^2.*fft(A2_xx,N_y-1,2),N_y-1,2);
gamma2(:,end) = gamma2(:,1);
gamma2(end,:) = gamma2(1,:);

ERR2 = get_L_x_inverse_per(gamma2, x, y, dx, dy, dt, kappa, beta_BDF);
ERR2 = (1/alpha^4)*get_L_y_inverse_per(ERR2, x, y, dx, dy, dt, kappa, beta_BDF);

A2(:,:,3) = A2(:,:,3) - ERR2;
% A2(:,end) = A2(:,1);
% A2(end,:) = A2(1,:);
