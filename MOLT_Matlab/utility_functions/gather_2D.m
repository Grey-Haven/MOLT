function F_p = gather_2D(F_mesh, x1_p, x2_p, x, y, dx, dy)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Gathers (uniform) mesh data to a single particle with coordinates (x1_p, x2_p).
    %
    % This function provides the mesh-to-particle mapping using linear splines.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Logical indices
    x_idx = 1 + floor( abs(x1_p - x(1))/dx );
    y_idx = 1 + floor( abs(x2_p - y(1))/dy );
    
    % Fraction of the cell relative to the left grid point in each direction
    f_x = (x1_p - x(x_idx))/dx;
    f_y = (x2_p - y(y_idx))/dy;

    % Distribute each field to the particle
    F_1 = F_mesh(x_idx, y_idx)*(1 - f_x)*(1 - f_y);
    F_2 = F_mesh(x_idx, y_idx+1)*(1 - f_x)*f_y;
    F_3 = F_mesh(x_idx+1, y_idx)*f_x*(1 - f_y);
    F_4 = F_mesh(x_idx+1, y_idx+1)*f_x*f_y;

    F_p = F_1 + F_2 + F_3 + F_4;
end