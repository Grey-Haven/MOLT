function F_mesh = scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
    %%%%%%%%
    % Scatters a single particle with coordinates (x1_p, x2_p) onto 
    % uniform mesh points with an area rule.
    %
    % This function uses linear splines to map particle data onto a mesh.
    %%%%%%%%
    
    % Logical indices
    x_idx = 1 + floor((x1_p - x(1))/dx);
    y_idx = 1 + floor((x2_p - y(1))/dy);
    
    % Fraction of the cell relative to the left grid point in each direction
    f_x = (x1_p - x(x_idx))/dx;
    f_y = (x2_p - y(y_idx))/dy;

    % Weight the particle info to the mesh
%     disp(x_idx + " " + y_idx)

    F_mesh = zeros(Nx,Ny);

    F_mesh(x_idx, y_idx)     = F_mesh(x_idx, y_idx)     + weight*(1 - f_x)*(1 - f_y);
    F_mesh(x_idx, y_idx+1)   = F_mesh(x_idx, y_idx+1)   + weight*(1 - f_x)*f_y;
    F_mesh(x_idx+1, y_idx)   = F_mesh(x_idx+1, y_idx)   + weight*f_x*(1 - f_y);
    F_mesh(x_idx+1, y_idx+1) = F_mesh(x_idx+1, y_idx+1) + weight*f_x*f_y;
end