function F_mesh = scatter_2D_vectorized_quadratic(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
    %%%%%%%%
    % Scatters a single particle with coordinates (x1_p, x2_p) onto 
    % uniform mesh points with an area rule.
    %
    % This function uses quadratic splines to map particle data onto a mesh.
    %
    % This assumes a grid of 
    % [a_x, b_x) x [a_y, b_y), that is, it does not include the right and
    % top boundaries.
    %%%%%%%%
    
    % Left/Bottom-most indices
    x_idx = 1 + floor((x1_p - x(1))/dx);
    y_idx = 1 + floor((x2_p - y(1))/dy);

    % If the particle is closer to the right/upper node, we want it to be
    % the center node.
    x_idx = x_idx + round((x1_p - x(x_idx))/dx);
    y_idx = y_idx + round((x2_p - y(y_idx))/dy);

    x_idx_m1 = x_idx - 1;
    x_idx_p1 = x_idx + 1;

    y_idx_m1 = y_idx - 1;
    y_idx_p1 = y_idx + 1;
    
    % Fraction of the cell relative to the left grid point in each direction
    fxs = (x1_p - x(x_idx))/dx;
    fys = (x2_p - y(y_idx))/dy;

    fxs_p1 = (x(x_idx) + dx - x1_p)/dx;
    fxs_m1 = (x(x_idx) - dx - x1_p)/dx;

    fys_p1 = (y(y_idx) + dy - x2_p)/dy;
    fys_m1 = (y(y_idx) - dy - x2_p)/dy;

    x_idx_m1 = mod(x_idx_m1 - 1, Nx) + 1;
    x_idx    = mod(x_idx    - 1, Nx) + 1;
    x_idx_p1 = mod(x_idx_p1 - 1, Nx) + 1;

    y_idx_m1 = mod(y_idx_m1 - 1, Ny) + 1;
    y_idx    = mod(y_idx    - 1, Ny) + 1;
    y_idx_p1 = mod(y_idx_p1 - 1, Ny) + 1;

    wxs_m1 = 1/2*(1/2 - fxs).^2;
    wxs    = 3/4 - fxs.^2;
    wxs_p1 = 1/2*(1/2 + fxs).^2;

    wys_m1 = 1/2*(1/2 - fys).^2;
    wys    = 3/4 - fys.^2;
    wys_p1 = 1/2*(1/2 + fys).^2;

    wxs_p1_alt = 1/8*(3 - 2*abs(fxs_p1)).^2;
    wxs_alt = 3/4 - fxs.^2;
    wxs_m1_alt = 1/8*(3 - 2*abs(fxs_m1)).^2;

    wys_p1_alt = 1/8*(3 - 2*abs(fys_p1)).^2;
    wys_alt = 3/4 - fys.^2;
    wys_m1_alt = 1/8*(3 - 2*abs(fys_m1)).^2;

    % assert(all(wxs_p1_alt == wxs_p1));
    % assert(all(wxs_alt == wxs));
    % assert(all(wxs_m1_alt == wxs_m1));
    % 
    % assert(all(wys_p1_alt == wys_p1));
    % assert(all(wys_alt == wys));
    % assert(all(wys_m1_alt == wys_m1));

    sz = [Ny,Nx];

    %
    % F7  F8  F9
    % F4  F5  F6
    % F1  F2  F3
    %    
    F1 = accumarray([y_idx_m1, x_idx_m1], wxs_m1 .* wys_m1 .* weight, sz);
    F2 = accumarray([y_idx_m1, x_idx   ], wxs    .* wys_m1 .* weight, sz);
    F3 = accumarray([y_idx_m1, x_idx_p1], wxs_p1 .* wys_m1 .* weight, sz);

    F4 = accumarray([y_idx   , x_idx_m1], wxs_m1 .* wys    .* weight, sz);
    F5 = accumarray([y_idx   , x_idx   ], wxs    .* wys    .* weight, sz);
    F6 = accumarray([y_idx   , x_idx_p1], wxs_p1 .* wys    .* weight, sz);

    F7 = accumarray([y_idx_p1, x_idx_m1], wxs_m1 .* wys_p1 .* weight, sz);
    F8 = accumarray([y_idx_p1, x_idx   ], wxs    .* wys_p1 .* weight, sz);
    F9 = accumarray([y_idx_p1, x_idx_p1], wxs_p1 .* wys_p1 .* weight, sz);

    F_mesh = F1+F2+F3+F4+F5+F6+F7+F8+F9;

    % F_mesh = zeros(Ny,Nx);
    % for p = 1:length(x1_p)
    %     F_mesh(y_idx_m1(p), x_idx_m1(p)) = F_mesh(y_idx_m1(p), x_idx_m1(p)) +  wxs_m1(p) * wys_m1(p) * weight(p);
    %     F_mesh(y_idx_m1(p), x_idx   (p)) = F_mesh(y_idx_m1(p), x_idx   (p)) +  wxs   (p) * wys_m1(p) * weight(p);
    %     F_mesh(y_idx_m1(p), x_idx_p1(p)) = F_mesh(y_idx_m1(p), x_idx_p1(p)) +  wxs_p1(p) * wys_m1(p) * weight(p);
    % 
    %     F_mesh(y_idx(p)   , x_idx_m1(p)) = F_mesh(y_idx   (p), x_idx_m1(p)) +  wxs_m1(p) * wys   (p) * weight(p);
    %     F_mesh(y_idx(p)   , x_idx   (p)) = F_mesh(y_idx   (p), x_idx   (p)) +  wxs   (p) * wys   (p) * weight(p);
    %     F_mesh(y_idx(p)   , x_idx_p1(p)) = F_mesh(y_idx   (p), x_idx_p1(p)) +  wxs_p1(p) * wys   (p) * weight(p);
    % 
    %     F_mesh(y_idx_p1(p), x_idx_m1(p)) = F_mesh(y_idx_p1(p), x_idx_m1(p)) +  wxs_m1(p) * wys_p1(p) * weight(p);
    %     F_mesh(y_idx_p1(p), x_idx   (p)) = F_mesh(y_idx_p1(p), x_idx   (p)) +  wxs   (p) * wys_p1(p) * weight(p);
    %     F_mesh(y_idx_p1(p), x_idx_p1(p)) = F_mesh(y_idx_p1(p), x_idx_p1(p)) +  wxs_p1(p) * wys_p1(p) * weight(p);
    % end
end