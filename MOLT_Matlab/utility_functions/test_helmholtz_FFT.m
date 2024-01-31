clear;
close all;

grids = [8,16,32,64,128,256];
errs = zeros(length(grids),1);
dxs = zeros(length(grids),1);
i = 1;

for g = grids
    N_x = g+1;
    N_y = g+1;
    
    a_x = -1;
    a_y = -1;
    b_x = 1;
    b_y = 1;
    
    L_x = b_x - a_x;
    L_y = b_y - a_y;
    
    dx = L_x / (N_x - 1);
    dy = L_y / (N_y - 1);
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;
    
    kx_deriv_2 = 2*pi/(L_x)*[0:(N_x-1)/2-1, -(N_x)/2, -(N_x-1)/2+1:-1];
    ky_deriv_2 = 2*pi/(L_y)*[0:(N_y-1)/2-1, -(N_y)/2, -(N_y-1)/2+1:-1];
    
    w_x = pi;
    w_y = pi;
    
    u_analytic = sin(w_x*x)'.*sin(w_y*y);
    
    % rough estimates of the problem
    kappa = 770;
    dt = 5e-5;
    
    alpha = sqrt(2) / (kappa*dt);
    % alpha = 1;
    
    S = (1 + 1/alpha^2 * (w_x^2 + w_y^2))*u_analytic;
    
    u_FFT = solve_helmholtz_FFT(S,alpha,kx_deriv_2,ky_deriv_2);

    errs(i) = max(max(abs(u_FFT - u_analytic)));
    dxs(i) = dx;
    i = i+1;
end

loglog(dxs,errs);