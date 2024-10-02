function F_p = gather_2D_vectorized_quadratic(F_mesh, x1_p, x2_p, x, y, dx, dy)
    %%%%%%%%
    % Gathers the field information corresponding to a single particle
    % with coordinates (x1_p, x2_p)
    %
    % This function uses quadratic splines to map mesh data to the particle
    %
    % Unlike the linear method, this assumes a grid of 
    % [a_x, b_x) x [a_y, b_y), that is, it does not include the right and
    % top boundaries.
    %%%%%%%%

    Nx = length(x);
    Ny = length(y);
                        
    index_offset = 1;

    % Left/Bottom-most indices
    x_idx = index_offset + floor((x1_p - x(1))/dx);
    y_idx = index_offset + floor((x2_p - y(1))/dy);

    % If the particle is closer to the right/upper node, we want it to be
    % the center node.
    x_idx = x_idx + round((x1_p - x(x_idx))/dx);
    y_idx = y_idx + round((x2_p - y(y_idx))/dy);

    x_idx_m1 = x_idx - 1;
    x_idx_p1 = x_idx + 1;

    y_idx_m1 = y_idx - 1;
    y_idx_p1 = y_idx + 1;

    x_idx_m1 = mod(x_idx_m1 - 1, Nx) + 1;
    x_idx    = mod(x_idx    - 1, Nx) + 1;
    x_idx_p1 = mod(x_idx_p1 - 1, Nx) + 1;

    y_idx_m1 = mod(y_idx_m1 - 1, Ny) + 1;
    y_idx    = mod(y_idx    - 1, Ny) + 1;
    y_idx_p1 = mod(y_idx_p1 - 1, Ny) + 1;

    %
    % idx7  idx8  idx9
    % idx4  idx5  idx6
    % idx1  idx2  idx3
    %

    idx1 = sub2ind(size(F_mesh), y_idx_m1, x_idx_m1);
    idx2 = sub2ind(size(F_mesh), y_idx_m1, x_idx   );
    idx3 = sub2ind(size(F_mesh), y_idx_m1, x_idx_p1);

    idx4 = sub2ind(size(F_mesh), y_idx   , x_idx_m1);
    idx5 = sub2ind(size(F_mesh), y_idx   , x_idx   );
    idx6 = sub2ind(size(F_mesh), y_idx   , x_idx_p1);

    idx7 = sub2ind(size(F_mesh), y_idx_p1, x_idx_m1);
    idx8 = sub2ind(size(F_mesh), y_idx_p1, x_idx   );
    idx9 = sub2ind(size(F_mesh), y_idx_p1, x_idx_p1);


    fxs = (x1_p - x(x_idx))/dx;
    fys = (x2_p - y(y_idx))/dy;

    wxs_m1 = 1/2*(1/2 - fxs).^2;
    wxs    = 3/4 - fxs.^2;
    wxs_p1 = 1/2*(1/2 + fxs).^2;

    wys_m1 = 1/2*(1/2 - fys).^2;
    wys    = 3/4 - fys.^2;
    wys_p1 = 1/2*(1/2 + fys).^2;


    %
    % F7  F8  F9
    % F4  F5  F6
    % F1  F2  F3
    %

    F1 = F_mesh(idx1) .* wys_m1.*wxs_m1;
    F2 = F_mesh(idx2) .* wys_m1.*wxs   ;
    F3 = F_mesh(idx3) .* wys_m1.*wxs_p1;

    F4 = F_mesh(idx4) .* wys   .*wxs_m1;
    F5 = F_mesh(idx5) .* wys   .*wxs   ;
    F6 = F_mesh(idx6) .* wys   .*wxs_p1;
   
    F7 = F_mesh(idx7) .* wys_p1.*wxs_m1;
    F8 = F_mesh(idx8) .* wys_p1.*wxs   ;
    F9 = F_mesh(idx9) .* wys_p1.*wxs_p1;

    F_p = F1+F2+F3+F4+F5+F6+F7+F8+F9;
end