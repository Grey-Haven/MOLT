clear;
% Using method of manufactured solutions

grids = [8,16,32,64,128,256];

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

    for i = 1:N_steps
        t = i*dt;
        u = analytic(x,y,t,Lx,Ly,alpha,beta,gamma);
        S = source(x,y,t,Lx,Ly,alpha,beta,gamma);
%         surf(x,y,u);
%         zlim([-1,1]);
%         drawnow;
    end

    u = zeros(Ny,Nx,3);
    u(:,:,1) = analytic(x,y,-dt,Lx,Ly,alpha,beta,gamma);
    u(:,:,2) = analytic(x,y,0,Lx,Ly,alpha,beta,gamma);

    ddx_u = zeros(Nx,Ny);
    ddy_u = zeros(Nx,Ny);
    S = zeros(Nx,Ny,2);

    for i = 1:N_steps
    
        v = (u(:,:,2) - u(:,:,1))/dt;

        t_n = i*dt;

        S_prev = source(x,y,t_n - dt,Lx,Ly,alpha,beta,gamma);
        S_curr = source(x,y,t_n     ,Lx,Ly,alpha,beta,gamma);
        S(:,:,1) = S_prev;
        S(:,:,2) = S_curr;

        [u_next, v_next] = DIRK2_advance_per(u(:,:,2), v, S, kappa, dt, kx_deriv_2, ky_deriv_2);
        u_next = real(u_next);
        u(:,:,end) = u_next;

        u(:,:,1) = u(:,:,2);
        u(:,:,2) = u(:,:,3);
    end

    t_n = i*dt;

    u_analytic = analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma);
    
    subplot(1,3,1);
    surf(x,y,u_next);
    title("$u_{DIRK}$",'interpreter','latex');
    subplot(1,3,2);
    surf(x,y,u_analytic);
    title("$u_{analytic}$",'interpreter','latex');
    subplot(1,3,3);
    surf(x,y,u_analytic - u_next);
    title("$u_{analytic} - u_{DIRK}$",'interpreter','latex');
    sgtitle(g + "x" + g);
    
%     diffs(r) = max(max(abs(u(:,:,3)) - analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma)));
    diffs(r) = norm(norm(u(:,:,3)) - analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma))/(Nx*Ny);
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