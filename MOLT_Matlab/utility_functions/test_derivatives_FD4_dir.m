clear;
close all;

grids = [8,16,32,64,128,256];
errs_x = zeros(length(grids),1);
errs_y = zeros(length(grids),1);
dxs = zeros(length(grids),1);
i = 1;

% x_eq = @(x_arg) x_arg + sin(x_arg);
x_eq = @(x_arg) x_arg.^3 - 3*x_arg + 1;
y_eq = @(y_arg) y_arg.^4 + y_arg.^2 + 2*y_arg;

x_deriv = @(x_arg) 3*x_arg.^2 - 3;
y_deriv = @(y_arg) 4*y_arg.^3 + 2*y_arg + 2;

for g = grids
    N_x = g+1;
    N_y = g+1;
    
    a_x = -10;
    a_y = -10;
    b_x =  10;
    b_y =  10;
    
    L_x = b_x - a_x;
    L_y = b_y - a_y;
    
    dx = L_x / (N_x - 1);
    dy = L_y / (N_y - 1);
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;
    
    u = (x_eq(x)'.*y_eq(y))';
    u_ddx = (x_deriv(x)'.*y_eq(y))';
    u_ddy = (x_eq(x)'.*y_deriv(y))';

    [u_ddx_fd4, u_ddy_fd4] = compute_gradient_FD4_dir(u,dx,dy);

    errs_x(i) = max(max(abs(u_ddx - u_ddx_fd4)));
    errs_y(i) = max(max(abs(u_ddy - u_ddy_fd4)));
    dxs(i) = dx;
    i = i+1;
end

loglog(dxs,errs_x);
hold on;
loglog(dxs,errs_y);
legend("||dudx - FD4_x[u]||", "||dudy - FD4_y[u]||");