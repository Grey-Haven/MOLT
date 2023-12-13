close all;
clear;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

% grids = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,256];
grids = 32:2:128;
errs = zeros(length(grids),1);
errs_nm1 = zeros(length(grids),1);

ddx_errs = zeros(length(grids),1);
ddx_errs_nm1 = zeros(length(grids),1);
ddx_errs_fft = zeros(length(grids),1);

T = .0001;

method = "BDF-1 + BDF-4 Hybrid";

for g_idx = 1:length(grids)
    N = grids(g_idx);

    N_x = N+1;
    N_y = N+1;

    disp(N + "x" + N);
    
    N_h = 6;
    
    psi = zeros(N_y,N_x,N_h);
    A1 = zeros(N_y,N_x,N_h);
    A2 = zeros(N_y,N_x,N_h);

    ddx_psi_fft = zeros(N_y,N_x);
    ddy_psi_fft = zeros(N_y,N_x);
    
    ddx_psi = zeros(N_y,N_x);
    ddy_psi = zeros(N_y,N_x);
    psi_src = zeros(N_y,N_x);
    
    ddx_A1 = zeros(N_y,N_x);
    ddy_A1 = zeros(N_y,N_x);
    A1_src = zeros(N_y,N_x);
    
    ddx_A2 = zeros(N_y,N_x);
    ddy_A2 = zeros(N_y,N_x);
    A2_src = zeros(N_y,N_x);
    
    L_x = 1.0;
    L_y = 1.0;
    
    a_x = -L_x/2;
    b_x = L_x/2;
    
    a_y = -L_y/2;
    b_y =  L_y/2;
    
    dx = (b_x - a_x)/(N_x - 1);
    dy = (b_y - a_y)/(N_y - 1);
    
    x = linspace(a_x, b_x, N_x);
    y = linspace(a_y, b_y, N_y);
    
    x_star = x(1:end-1);
    y_star = y(1:end-1);
    
    kx_deriv_1 = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];
    ky_deriv_1 = 2*pi/(L_y)*[0:(N_y-1)/2-1, 0, -(N_y-1)/2+1:-1];
    
    kx_deriv_2 = 2*pi/(L_x)*[0:(N_x-1)/2-1, -(N_x)/2, -(N_x-1)/2+1:-1];
    ky_deriv_2 = 2*pi/(L_y)*[0:(N_y-1)/2-1, -(N_y)/2, -(N_y-1)/2+1:-1];
    
    beta_BDF = 1;
    
    xi_x = 2*pi;
    xi_y = 2*pi;
    xi_t = 2*pi;
    % xi_t = sqrt(xi_x^2 + xi_y^2);
    kappa = 770;
    
    % test_func = sin(xi_y*y_star)'.*sin(xi_x*x_star);
    % test_func_xx = -xi_x^2*sin(xi_y*y_star)'.*sin(xi_x*x_star);
    % test_func_yy = -xi_y^2*sin(xi_y*y_star)'.*sin(xi_x*x_star);
    % 
    % test_func_fft_xx = ifft(-kx_deriv_2.^2.*fft(test_func,N_x-1,2),N_x-1,2);
    % test_func_fft_yy = ifft(-ky_deriv_2'.^2.*fft(test_func,N_y-1,1),N_y-1,1);
    % 
    % assert(norm(test_func_fft_xx - test_func_xx) < 1e-9);
    
    t = 0;
    dt = dx/(sqrt(2)*kappa);
    
    for idx = 1:N_h-1
        dt_offset = (N_h - idx - 1)*dt;
        psi(:,:,idx) = u(x,y,t-dt_offset,xi_x,xi_y,xi_t,kappa);
        A1(:,:,idx) = u(x,y,t-dt_offset,xi_x,xi_y,xi_t,kappa);
        A2(:,:,idx) = u(x,y,t-dt_offset,xi_x,xi_y,xi_t,kappa);
    end
    
    t = dt;
    
    psi_err = zeros(floor(T/dt),1);
    A1_err = zeros(floor(T/dt),1);
    A2_err = zeros(floor(T/dt),1);
    
    step = 1;
    plot_at = 1;
    
    psi_src = zeros(size(psi(:,:,1)));
    A1_src = zeros(size(psi(:,:,1)));
    A2_src = zeros(size(psi(:,:,1)));
    
    % vidName = "/test_results" + ".mp4";
    % vidObj = VideoWriter(pwd + vidName, 'MPEG-4');
    % open(vidObj);
    
    % while t < T
    S_fin = 1;
    for s = 1:S_fin
        t = dt*s;
        t_n = t;
    
        % psi_src = S(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
        % A1_src = S(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
        % A2_src = S(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
    
        % update_waves;
        % update_waves_FFT;
        % update_waves_FFT_alt;
        update_waves_hybrid_BDF;
    
        psi = shuffle_steps(psi);
        A1 = shuffle_steps(A1);
        A2 = shuffle_steps(A2);

        analytic_nm2 = u(x,y,t-2*dt,xi_x,xi_y,xi_t,kappa);
        analytic_nm1 = u(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
        analytic = u(x,y,t,xi_x,xi_y,xi_t,kappa);
        analytic_np1 = u(x,y,t+dt,xi_x,xi_y,xi_t,kappa);

        ddx_analytic_nm1 = u_x(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
        ddx_analytic = u_x(x,y,t,xi_x,xi_y,xi_t,kappa);

        psi_next = psi(1:end-1,1:end-1,end);

        psi_next_fft_x = fft(psi_next,N_x-1,2);
        psi_next_fft_y = fft(psi_next,N_y-1,1);
        
        ddx_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .* psi_next_fft_x,N_x-1,2);
        ddy_psi_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.* psi_next_fft_y,N_y-1,1);
        
        ddx_psi_fft = copy_periodic_boundaries(ddx_psi_fft);
        ddy_psi_fft = copy_periodic_boundaries(ddy_psi_fft);
        
    %     if mod(step,plot_at) == 0
    % 
    %         subplot(1,3,1);
    %         surf(x,y,psi(:,:,3));
    %         xlabel("x");
    %         ylabel("y");
    %         title("Approx Psi");
    %         zlim([-1,1]);
    % 
    %         subplot(1,3,2);
    %         surf(x,y,u(x,y,t,xi_x,xi_y,xi_t,kappa));
    %         xlabel("x");
    %         ylabel("y");
    %         title("Analytic");
    %         zlim([-1,1]);
    % 
    %         subplot(1,3,3);
    %         surf(x,y,u(x,y,t,xi_x,xi_y,xi_t,kappa) - psi(:,:,end));
    %         xlabel("x");
    %         ylabel("y");
    %         title("Analytic - Approx Psi");
    % %         zlim([-1e-2,1e-2]);
    % 
    %         sgtitle("t = " + t);
    %         drawnow;
    %         currFrame = getframe(gcf);
    %         writeVideo(vidObj, currFrame);
    %     end
    
        % psi_err(step) = max(max(abs(analytic - psi(:,:,end))));
        % A1_err(step) = max(max(abs(analytic - A1(:,:,end))));
        % A2_err(step) = max(max(abs(analytic - A2(:,:,end))));
        % 
        t = t + dt;
        % step = step + 1;
    end
    % close(vidObj);
    
    % errs_nm1(g_idx) = max(max(abs(analytic_nm1 - psi(:,:,end))));
    % ddx_errs_nm1(g_idx) = max(max(abs(ddx_analytic_nm1 - ddx_psi(:,:,end))));
    % 
    % errs(g_idx) = max(max(abs(analytic - psi(:,:,end))));
    % ddx_errs(g_idx) = max(max(abs(ddx_analytic- ddx_psi(:,:,end))));
    % 
    % ddx_errs_fft(g_idx) = max(max(abs(ddx_analytic - ddx_psi_fft)));

    errs(g_idx) = get_L_2_error(analytic - psi(:,:,end), ...
                                zeros(size(analytic(:,:))), dx*dy);
    ddx_errs(g_idx) = get_L_2_error(ddx_analytic- ddx_psi(:,:,end), ...
                                    zeros(size(analytic(:,:))), dx*dy);
    ddx_errs_fft(g_idx) = get_L_2_error(ddx_analytic- ddx_psi_fft(:,:,end), ...
                                        zeros(size(analytic(:,:))), dx*dy);
    % ts = 0:dt:T-dt;
    % 
    % figure;
    % plot(ts,psi_err);
    % hold on;
    % plot(ts,A1_err);
    % plot(ts,A2_err);
    % xlabel("t");
    % ylabel("Inf Err");
    % legend("Psi", "A_1", "A_2");
end

dxs = L_x ./ grids;

linear = @(x_arg) x_arg;
quad = @(x_arg) x_arg.^2;
cubic = @(x_arg) x_arg.^3;
quartic = @(x_arg) x_arg.^4;
quintic = @(x_arg) x_arg.^5;

figure;

subplot(1,2,1);
loglog(dxs,errs);
hold on;
loglog(dxs,linear(dxs));
loglog(dxs,quad(dxs));
loglog(dxs,cubic(dxs));
loglog(dxs,quartic(dxs));
loglog(dxs,quintic(dxs));
legend("Error", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Location", "SouthEast");
xlabel("dx");
ylabel("Max Err")
title("Errors for Wave");

subplot(1,2,2);

loglog(dxs,ddx_errs,"--");
hold on;
loglog(dxs,ddx_errs_fft,"-.");
loglog(dxs,linear(dxs));
loglog(dxs,quad(dxs));
loglog(dxs,cubic(dxs));
loglog(dxs,quartic(dxs));
loglog(dxs,quintic(dxs));
legend("Error", "FFT Error", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Location", "SouthEast");
xlabel("dx");
ylabel("Max Err")
title("Errors for Derivative");

sgtitle(method);

function src = S(x,y,t, xi_x,xi_y,xi_t,k)
    src = -(xi_x^2 - xi_t^2) * u(x,y,t,xi_x,xi_y,xi_t,k);
end

function analytic = u(x,y,t, xi_x,xi_y,xi_t,k)
    analytic = ones(length(y),1) .* cos(xi_x*x) * cos(k*xi_t*t);
    % analytic = zeros(length(y),length(x));
end

function analytic_ddx = u_x(x,y,t, xi_x,xi_y,xi_t,k)
    analytic_ddx = -xi_x*ones(length(y),1) .* sin(xi_x*x) * cos(k*xi_t*t);
    % analytic = zeros(length(y),length(x));
end

% function src = S(x,y,t, xi_x,xi_y,xi_t,k)
%     src = -(xi_x^2 + xi_y^2 - xi_t^2) * u(x,y,t,xi_x,xi_y,xi_t,k);
% end
% 
% function analytic = u(x,y,t, xi_x,xi_y,xi_t,k)
%     analytic = cos(xi_y*y)' .* cos(xi_x*x) * sin(k*xi_t*t);
% end