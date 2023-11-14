function F_mesh = scatter_2D(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
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

    F_mesh = zeros(Ny,Nx);

    F_mesh(y_idx,     x_idx) = F_mesh(y_idx,     x_idx) + weight*(1 - f_y)*(1 - f_x);
    F_mesh(y_idx,   x_idx+1) = F_mesh(y_idx,   x_idx+1) + weight*(1 - f_y)*f_x;
    F_mesh(y_idx+1,   x_idx) = F_mesh(y_idx+1,   x_idx) + weight*f_y*(1 - f_x);
    F_mesh(y_idx+1, x_idx+1) = F_mesh(y_idx+1, x_idx+1) + weight*f_y*f_x;
end