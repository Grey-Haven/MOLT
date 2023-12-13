% Compute splitting error with FFT
% gamma = (1/alpha^4)(dxx)(dyy){A1,A2}

alpha = beta_BDF/(kappa*dt);

gamma1 = zeros(size(A1 (:,:,3)));
gamma2 = zeros(size(A2 (:,:,3)));
gamma3 = zeros(size(psi(:,:,3)));

A1_curr  = A1 (1:end-1,1:end-1,3);
A2_curr  = A2 (1:end-1,1:end-1,3);
psi_curr = psi(1:end-1,1:end-1,3);

A1_xx = ifft(-kx_deriv_2'.^2.*fft(A1_curr,N_x-1,1),N_x-1);
gamma1(1:end-1,1:end-1) = ifft(-ky_deriv_2.^2.*fft(A1_xx,N_y-1,2),N_y-1,2);
gamma1(:,end) = gamma1(:,1);
gamma1(end,:) = gamma1(1,:);


ERR1 = get_L_y_inverse_per(gamma1, x, y, dx, dy, dt, kappa, beta_BDF);
ERR1 = (1/alpha^4)*get_L_x_inverse_per(ERR1, x, y, dx, dy, dt, kappa, beta_BDF);

uncleaned1 = A1(:,:,3);
A1(:,:,3) = A1(:,:,3) - ERR1;
% A1(:,end) = A1(:,1);
% A1(end,:) = A1(1,:);

A2_xx = ifft(-kx_deriv_2'.^2.*fft(A2_curr,N_x-1,1),N_x-1);
gamma2(1:end-1,1:end-1) = ifft(-ky_deriv_2.^2.*fft(A2_xx,N_y-1,2),N_y-1,2);
gamma2(:,end) = gamma2(:,1);
gamma2(end,:) = gamma2(1,:);

ERR2 = get_L_y_inverse_per(gamma2, x, y, dx, dy, dt, kappa, beta_BDF);
ERR2 = (1/alpha^4)*get_L_x_inverse_per(ERR2, x, y, dx, dy, dt, kappa, beta_BDF);

uncleaned2 = A2(:,:,3);

A2(:,:,3) = A2(:,:,3) - ERR2;

psi_xx = ifft(-kx_deriv_2'.^2.*fft(psi_curr,N_x-1,1),N_x-1);
gamma3(1:end-1,1:end-1) = ifft(-ky_deriv_2.^2.*fft(psi_xx,N_y-1,2),N_y-1,2);
gamma3(:,end) = gamma3(:,1);
gamma3(end,:) = gamma3(1,:);

ERR3 = get_L_y_inverse_per(gamma3, x, y, dx, dy, dt, kappa, beta_BDF);
ERR3 = (1/alpha^4)*get_L_x_inverse_per(ERR3, x, y, dx, dy, dt, kappa, beta_BDF);

uncleaned3 = psi(:,:,3);

psi(:,:,3) = psi(:,:,3) - ERR3;


% subplot(3,3,1)
% surf(x,y,uncleaned1)
% title("Uncleaned A1")
% xlabel("x")
% ylabel("y")
% subplot(3,3,4)
% surf(x,y,A1(:,:,3))
% title("Cleaned A1")
% xlabel("x");
% ylabel("y");
% subplot(3,3,7);
% surf(x,y,ERR1);
% title("ERR");
% xlabel("x");
% ylabel("y");
% 
% subplot(3,3,2)
% surf(x,y,uncleaned2)
% title("Uncleaned A2")
% xlabel("x")
% ylabel("y")
% subplot(3,3,5)
% surf(x,y,A2(:,:,3))
% title("Cleaned A2")
% xlabel("x");
% ylabel("y");
% subplot(3,3,8);
% surf(x,y,ERR2);
% title("ERR");
% xlabel("x");
% ylabel("y");
% 
% subplot(3,3,3)
% surf(x,y,uncleaned3)
% title("Uncleaned phi")
% xlabel("x")
% ylabel("y")
% subplot(3,3,6)
% surf(x,y,psi(:,:,3))
% title("Cleaned phi")
% xlabel("x");
% ylabel("y");
% subplot(3,3,9);
% surf(x,y,ERR3);
% title("ERR");
% xlabel("x");
% ylabel("y");
% 
% sgtitle("Vanilla method, t = " + t_n);
% drawnow;
% currFrame = getframe(gcf);
% writeVideo(vidObj, currFrame);
