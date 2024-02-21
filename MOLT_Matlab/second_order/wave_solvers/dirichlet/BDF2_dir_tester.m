clear;
% close all;

grids = [16,32,64,128,256];

diffs = zeros(length(grids),1);
dxs = zeros(length(grids),1);

dx_min = 2 / max(grids);
dt_min = dx_min / 6;

for r = 1:length(grids)

    g = grids(r);
    disp(g + "x" + g);

    Nx = g + 1;
    Ny = g + 1;

    N_steps = 1;
    
    a_x = -1;
    b_x = 1;
    
    a_y = -1;
    b_y = 1;
    
    Lx = b_x - a_x;
    Ly = b_y - a_y;
    
    alpha = 2*pi;
    beta = 2*pi;
    gamma = 1*pi;
    kappa = 1;

    beta_BDF = 1;
    
    dx = Lx / g;
    dy = Ly / g;

    dt = dx/2;
%     dt = dt_min;
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;

    kx_deriv_1 = 2*pi/(Lx)*[0:(Nx-1)/2-1, 0, -(Nx-1)/2+1:-1];
    ky_deriv_1 = 2*pi/(Ly)*[0:(Ny-1)/2-1, 0, -(Ny-1)/2+1:-1];
    
    kx_deriv_2 = 2*pi/(Lx)*[0:(Nx-1)/2-1, -(Nx)/2, -(Nx-1)/2+1:-1];
    ky_deriv_2 = 2*pi/(Ly)*[0:(Ny-1)/2-1, -(Ny)/2, -(Ny-1)/2+1:-1];

    u = zeros(Ny,Nx);
    ddx_u = zeros(Nx,Ny);
    ddy_u = zeros(Nx,Ny);
    S = zeros(Nx,Ny);

    % for i = 1:N_steps
    %     t = i*dt;
    %     u(:,:,i) = analytic(x,y,t,Lx,Ly,alpha,beta,gamma);
    % end

    u = zeros(Ny,Nx,6);
    u(:,:,1) = analytic(x,y,-4*dt,Lx,Ly,alpha,beta,gamma);
    u(:,:,2) = analytic(x,y,-3*dt,Lx,Ly,alpha,beta,gamma);
    u(:,:,3) = analytic(x,y,-2*dt,Lx,Ly,alpha,beta,gamma);
    u(:,:,4) = analytic(x,y,-1*dt,Lx,Ly,alpha,beta,gamma);
    u(:,:,5) = analytic(x,y, 0*dt,Lx,Ly,alpha,beta,gamma);

    for i = 1:N_steps
        t_n = i*dt;
        S(:,:) = source(x,y,t_n,Lx,Ly,alpha,beta,gamma);
        [u, ddx_u, ddy_u] = BDF2_combined_dir_advance(u, ddx_u, ddy_u, S(:,:), x, y, dx, dy, dt, kappa, beta_BDF);

%         surf(x,y,squeeze(u(3,:,:)));
% %         plot(x,squeeze(u(3,8,:)));
%         zlim([-1,1]);
%         drawnow;

%         u(:,:,1) = u(:,:,2);
%         u(:,:,2) = u(:,:,3);
    end

    u_analytic = analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma);
    
    laplacian_u_FFT = compute_Laplacian_FFT(u(:,:,end),kx_deriv_2,ky_deriv_2);
    ddt2_u = (u(:,:,end) - 2*u(:,:,end-1) + u(:,:,end-2))/(dt^2);
    LHS = 1/kappa^2*ddt2_u - laplacian_u_FFT;
    subplot(1,3,1);
    surf(x,y,u(:,:,end));
    title("u_{BDF2}");
    subplot(1,3,2);
    surf(x,y,u_analytic);
    title("u_{analytic}");
    subplot(1,3,3);
    surf(x,y,u(:,:,end) - u_analytic);
    title("u_{BDF2} - u_{analytic}");

    sgtitle(g + "x" + g);
    drawnow;
    
%     diffs(r) = max(max(abs(u(:,:,3)) - analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma)));
    diffs(r) = norm(norm(u(:,:,end)) - u_analytic)/(Nx*Ny);
    dxs(r) = dx;
end
figure;
plot(dxs,diffs);
xlabel("dx");
ylabel("l2 norm");


function u = analytic(x,y,t,Lx,Ly,alpha,beta,gamma)
    Nx = length(x);
    Ny = length(y);
    u = zeros(Nx,Ny);
    for i = 1:length(x)
        x_i = x(i);
        for j = 1:length(y)
            y_j = y(j);
            u(j,i) = sin((alpha/2)*x_i)*sin((beta/2)*y_j)*sin(gamma*t);
        end
    end
end


function S = source(x,y,t,Lx,Ly,alpha,beta,gamma)
    S = -(gamma^2 - alpha^2/4 - beta^2/4)*analytic(x,y,t,Lx,Ly,alpha,beta,gamma);
end