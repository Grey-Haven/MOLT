clear;
% Using method of manufactured solutions

grids = [8,16,32,64,128,256];

diffs = zeros(length(grids),1);
dxs = zeros(length(grids),1);

for r = 1:length(grids)

    g = grids(r);
    disp(g + "x" + g);

    Nx = g + 1;
    Ny = g + 1;

    N_steps = 100;
    
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
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;

    for i = 1:N_steps
        t = i*dt;
        u = analytic(x,y,t,Lx,Ly,alpha,beta,gamma);
        S = source(x,y,t,Lx,Ly,alpha,beta,gamma);
%         surf(x,y,u);
%         zlim([-1,1]);
%         drawnow;
    end

    u = zeros(3,Nx,Ny);
    u(1,:,:) = analytic(x,y,-dt,Lx,Ly,alpha,beta,gamma);
    u(2,:,:) = analytic(x,y,0,Lx,Ly,alpha,beta,gamma);
    
    ddx_u = zeros(Nx,Ny);
    ddy_u = zeros(Nx,Ny);
    S = zeros(Nx,Ny);

    for i = 1:N_steps
        t_n = i*dt;
        S(:,:) = source(x,y,t_n,Lx,Ly,alpha,beta,gamma);
        [u, ddx_u, ddy_u] = BDF1_combined_per_advance(u, ddx_u, ddy_u, S(:,:), x, y, t_n, dx, dy, dt, kappa, beta_BDF);

%         surf(x,y,squeeze(u(3,:,:)));
% %         plot(x,squeeze(u(3,8,:)));
%         zlim([-1,1]);
%         drawnow;

        u(1,:,:) = u(2,:,:);
        u(2,:,:) = u(3,:,:);
    end
    diffs(r) = norm(norm(squeeze(u(3,:,:)) - analytic(x,y,t_n,Lx,Ly,alpha,beta,gamma)))/(Nx*Ny);
    dxs(r) = dx;
end

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
            u(i,j) = sin((alpha/2)*x_i)*sin((beta/2)*y_j)*sin(gamma*t);
        end
    end
end


function S = source(x,y,t,Lx,Ly,alpha,beta,gamma)
    S = -(gamma^2 - alpha^2/4 - beta^2/4)*analytic(x,y,t,Lx,Ly,alpha,beta,gamma);
end