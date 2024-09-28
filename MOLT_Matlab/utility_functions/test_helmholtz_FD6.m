clear;
close all;

grids = [16,32,64,128,256];
errs = zeros(length(grids),1);
dxs = zeros(length(grids),1);
i = 1;

MAX = 10000;
TOL = 1e-15;

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
    
    x = a_x:dx:b_x-dx;
    y = a_y:dy:b_y-dy;
    
    w_x = 4*pi/L_x;
    w_y = 2*pi/L_y;
    
    u_analytic = sin(w_x*x)'.*sin(w_y*y);
    laplacian_analytic = -(w_x^2 + w_y^2)*u_analytic;
    
    % rough estimates of the problem
    kappa = 770;
    dt = 5e-5;
    
    alpha = sqrt(2) / (kappa*dt);
    % alpha = 1;
    
    u_guess = zeros(length(y), length(x));

    S = (1 + 1/alpha^2 * (w_x^2 + w_y^2))*u_analytic;

    laplacian = compute_Laplacian_FD6(u_analytic, dx, dy);
    
    % u_FD6 = solve_helmholtz_GS_FD6(u_guess, S, alpha, dx, dy, MAX, TOL);
    u_FD6 = solve_helmholtz_MG_FD6(u_guess, S, alpha, dx, dy);
    % u_FD6 = solve_helmholtz_FD6(S, alpha, dx, dy);

    subplot(1,3,1);
    surf(x,y,u_analytic);
    subplot(1,3,2);
    surf(x,y,u_FD6);
    subplot(1,3,3);
    surf(x,y,u_FD6 - u_analytic);
    drawnow;

    errs(i) = max(max(abs(u_FD6 - u_analytic)));
    dxs(i) = dx;
    i = i+1;
end

figure;
semilogy(dxs,errs);
title("Errors");