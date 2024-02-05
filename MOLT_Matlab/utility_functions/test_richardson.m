dts = 2.^[-1,-2,-3,-4,-5,-6,-7,-8];
gs = 2.^[4,5,6,7,8];
errs = zeros(length(dts),length(gs));

idx = 1;

for i = 1:length(dts)
    for j = 1:length(gs)
        g = gs(j);
        dt = dts(i);
        
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
        
        t_n = dt;
        
        kappa = 1;
        
        u_analytic = sin(pi*x)'.*sin(pi*y).*cos(t_n);
        
        u_0 = sin(pi*x)'.*sin(pi*y);
        u_approx = richardson_extrapolation(u_0,0,kappa,dt,kx_deriv_2,ky_deriv_2);
    
        errs(i,j) = norm(u_approx - u_analytic);
        idx = idx+1;
    end
end

surf(1./gs,dts,errs);
xlabel("dx");
ylabel("dt");

% subplot(1,3,1);
% surf(x,y,u_analytic);
% subplot(1,3,2);
% surf(x,y,u_approx);
% subplot(1,3,3);
% surf(x,y,u_approx-u_analytic);