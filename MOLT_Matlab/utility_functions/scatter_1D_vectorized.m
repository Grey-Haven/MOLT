function F_mesh = scatter_1D_vectorized(Nx, x1_p, x, dx, weight)
    %%%%%%%%
    % Scatters a single particle with coordinates (x1_p, x2_p) onto 
    % uniform mesh points with an area rule.
    %
    % This function uses linear splines to map particle data onto a mesh.
    %%%%%%%%
    
    % Logical indices
    x_idx = 1 + floor((x1_p - x(1))/dx);
    
    % Fraction of the cell relative to the left grid point in each direction
    fxs = (x1_p - x(x_idx))./dx;

    sz = [Nx, 1];
    
    F1 = accumarray([x_idx  ],(1-fxs).*weight,sz);
    F2 = accumarray([x_idx+1],   fxs .*weight,sz);

    F_mesh = F1+F2;
end