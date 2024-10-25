function F_mesh = scatter_2D_vectorized_linear(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
    %%%%%%%%
    % Scatters a single particle with coordinates (x1_p, x2_p) onto 
    % uniform mesh points with an area rule.
    %
    % This function uses linear splines to map particle data onto a mesh.
    %
    % This assumes a grid of 
    % [a_x, b_x) x [a_y, b_y), that is, it does not include the right and
    % top boundaries.
    %%%%%%%%
    
    % Logical indices
    x_idx = 1 + floor((x1_p - x(1))/dx);
    y_idx = 1 + floor((x2_p - y(1))/dy);
    
    % Fraction of the cell relative to the left grid point in each direction
    fxs = (x1_p - x(x_idx))/dx;
    fys = (x2_p - y(y_idx))/dy;

    x_idx_left    = x_idx;
    x_idx_rite    = x_idx + 1;

    y_idx_left    = y_idx;
    y_idx_rite    = y_idx + 1;

    % The -1 comes from the 1-indexing of Matlab
    x_idx_rite = mod(x_idx_rite - 1, Nx) + 1;
    y_idx_rite = mod(y_idx_rite - 1, Ny) + 1;

    sz = [Ny,Nx];
    
    F1 = accumarray([y_idx_left, x_idx_left],(1-fys).*(1-fxs).*weight,sz);
    F2 = accumarray([y_idx_left, x_idx_rite],(1-fys).*   fxs .*weight,sz);
    F3 = accumarray([y_idx_rite, x_idx_left],   fys .*(1-fxs).*weight,sz);
    F4 = accumarray([y_idx_rite, x_idx_rite],   fys .*   fxs .*weight,sz);

    F_mesh = F1+F2+F3+F4;
end