clear;
close all;

grids = [8,16,32,64];
errs_x = zeros(length(grids),1);
errs_y = zeros(length(grids),1);
dxs = zeros(length(grids),1);
dys = zeros(length(grids),1);
i = 1;
    
a_x = -1;
a_y = -1;
b_x =  1;
b_y =  1;

L_x = b_x - a_x;
L_y = b_y - a_y;

w_y = 2*pi/L_x;
w_x = 4*pi/L_y;

x_eq = @(x_arg) sin(w_x*x_arg);
y_eq = @(y_arg) cos(w_y*y_arg);

x_deriv = @(x_arg)  w_x*cos(w_x*x_arg);
y_deriv = @(y_arg) -w_y*sin(w_y*y_arg);

for g = grids
    N_x = g+1;
    N_y = g+1;
    
    L_x = b_x - a_x;
    L_y = b_y - a_y;
    
    dx = L_x / (N_x - 1);
    dy = L_y / (N_y - 1);
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;

    % for i = 1:N_x
    %     for j = 1:N_y
    %         u(j,i) = x_eq(x(i))*y_eq(y(j));
    %         u_ddx = x_deriv(x(i))*y(y(j));
    %         u_ddy = x_eq(x(i))*y_deriv(y(j));
    %     end
    % end
    
    u = y_eq(y)'.*x_eq(x);
    u_ddx = y_eq(y)'.*x_deriv(x);
    u_ddy = y_deriv(y)'.*x_eq(x);

    u_ddx_FD6 = compute_ddx_FD6_per(u(1:end-1,1:end-1), dx);
    u_ddy_FD6 = compute_ddy_FD6_per(u(1:end-1,1:end-1), dy);

    errs_x(i) = max(max(abs(u_ddx(1:end-1,1:end-1) - u_ddx_FD6)));
    errs_y(i) = max(max(abs(u_ddy(1:end-1,1:end-1) - u_ddy_FD6)));
    dxs(i) = dx;
    i = i+1;
end

loglog(dxs,errs_x);
loglog(dys,errs_y);
hold on;
loglog(dxs,dxs);
loglog(dxs,dxs.^2);
loglog(dxs,dxs.^3);
loglog(dxs,dxs.^4);
loglog(dxs,dxs.^5);
loglog(dxs,dxs.^6);
legend(["Errors", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Sextic"])