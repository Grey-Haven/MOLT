function F_mesh = scatter_2D_vectorized(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
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
    fxs = (x1_p - x(x_idx))/dx;
    fys = (x2_p - y(y_idx))/dy;

    sz = [Ny,Nx];
    
    F1 = accumarray([y_idx,x_idx],(1-fys).*(1-fxs).*weight,sz);
    F2 = accumarray([y_idx,x_idx+1],(1-fys).*fxs.*weight,sz);
    F3 = accumarray([y_idx+1,x_idx],fys.*(1-fxs).*weight,sz);
    F4 = accumarray([y_idx+1,x_idx+1],fys.*fxs.*weight,sz);

    F_mesh = F1+F2+F3+F4;
end