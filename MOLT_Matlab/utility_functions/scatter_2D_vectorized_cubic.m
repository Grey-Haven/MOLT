function F_mesh = scatter_2D_vectorized_cubic(Nx, Ny, x1_p, x2_p, x, y, dx, dy, weight)
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
    x_idx_left = 1 + floor((x1_p - x(1))/dx);
    y_idx_left = 1 + floor((x2_p - y(1))/dy);

    x_idx_left_m1 = x_idx_left - 1;
    x_idx_rite = x_idx_left + 1;
    x_idx_rite_p1 = x_idx_left + 2;

    y_idx_left_m1 = y_idx_left - 1;
    y_idx_rite = y_idx_left + 1;
    y_idx_rite_p1 = y_idx_left + 2;

    % With cubic, like linear, the nodes are the four closest to the
    % particle
    % ie 
    %
    % N_l2      N_l1 p    N_r1      N_r2
    %

    x_left_m1 = x(x_idx_left) - dx;
    x_left    = x(x_idx_left);
    x_rite    = x(x_idx_left) + dx;
    x_rite_p1 = x(x_idx_left) + 2*dx;

    y_left_m1 = y(y_idx_left) - dy;
    y_left    = y(y_idx_left);
    y_rite    = y(y_idx_left) + dy;
    y_rite_p1 = y(y_idx_left) + 2*dy;
    
    % Fraction of the cell relative to the left grid point in each direction
    % fxs = (x1_p - x(x_idx_left))/dx;
    % fys = (x2_p - y(y_idx_left))/dy;
    fxs_left_m1 = (x_left_m1 - x1_p) / dx;
    fxs_left    = (x_left    - x1_p) / dx;
    fxs_rite    = (x_rite    - x1_p) / dx;
    fxs_rite_p1 = (x_rite_p1 - x1_p) / dx;

    fys_left_m1 = (y_left_m1 - x2_p) / dy;
    fys_left    = (y_left    - x2_p) / dy;
    fys_rite    = (y_rite    - x2_p) / dy;
    fys_rite_p1 = (y_rite_p1 - x2_p) / dy;

    % assert(all(abs(fxs_left_m1) < 2+1e-14))
    % assert(all(abs(fxs_left) < 1+1e-14));
    % assert(all(abs(fxs_rite) < 1+1e-14));
    % assert(all(abs(fxs_rite_p1) < 2+1e-14))

    center = @(x_arg) 2/3 - x_arg.^2 + abs(x_arg).^3 / 2;
    off    = @(x_arg) 1/6 * (2 - abs(x_arg)).^3;

    wxs_left_m1 = off(fxs_left_m1);
    wxs_left    = center(fxs_left);
    wxs_rite    = center(fxs_rite);
    wxs_rite_p1 = off(fxs_rite_p1);

    wys_left_m1 = off(fys_left_m1);
    wys_left    = center(fys_left);
    wys_rite    = center(fys_rite);
    wys_rite_p1 = off(fys_rite_p1);

    x_idx_left_m1 = mod(x_idx_left_m1 - 1, Nx) + 1;
    x_idx_left    = mod(x_idx_left    - 1, Nx) + 1;
    x_idx_rite    = mod(x_idx_rite    - 1, Nx) + 1;
    x_idx_rite_p1 = mod(x_idx_rite_p1 - 1, Nx) + 1;

    y_idx_left_m1 = mod(y_idx_left_m1 - 1, Ny) + 1;
    y_idx_left    = mod(y_idx_left    - 1, Ny) + 1;
    y_idx_rite    = mod(y_idx_rite    - 1, Ny) + 1;
    y_idx_rite_p1 = mod(y_idx_rite_p1 - 1, Ny) + 1;

    sz = [Ny,Nx];

    %
    % F13  F14  F15  F16
    % F09  F10  F11  F12
    % F05  F06  F07  F08
    % F01  F02  F03  F04
    %

    F01 = accumarray([y_idx_left_m1, x_idx_left_m1], wxs_left_m1 .* wys_left_m1 .* weight, sz);
    F02 = accumarray([y_idx_left_m1, x_idx_left   ], wxs_left    .* wys_left_m1 .* weight, sz);
    F03 = accumarray([y_idx_left_m1, x_idx_rite   ], wxs_rite    .* wys_left_m1 .* weight, sz);
    F04 = accumarray([y_idx_left_m1, x_idx_rite_p1], wxs_rite_p1 .* wys_left_m1 .* weight, sz);
    
    F05 = accumarray([y_idx_left   , x_idx_left_m1], wxs_left_m1 .* wys_left    .* weight, sz);
    F06 = accumarray([y_idx_left   , x_idx_left   ], wxs_left    .* wys_left    .* weight, sz);
    F07 = accumarray([y_idx_left   , x_idx_rite   ], wxs_rite    .* wys_left    .* weight, sz);
    F08 = accumarray([y_idx_left   , x_idx_rite_p1], wxs_rite_p1 .* wys_left    .* weight, sz);
    
    F09 = accumarray([y_idx_rite   , x_idx_left_m1], wxs_left_m1 .* wys_rite    .* weight, sz);
    F10 = accumarray([y_idx_rite   , x_idx_left   ], wxs_left    .* wys_rite    .* weight, sz);
    F11 = accumarray([y_idx_rite   , x_idx_rite   ], wxs_rite    .* wys_rite    .* weight, sz);
    F12 = accumarray([y_idx_rite   , x_idx_rite_p1], wxs_rite_p1 .* wys_rite    .* weight, sz);
      
    F13 = accumarray([y_idx_rite_p1, x_idx_left_m1], wxs_left_m1 .* wys_rite_p1 .* weight, sz);
    F14 = accumarray([y_idx_rite_p1, x_idx_left   ], wxs_left    .* wys_rite_p1 .* weight, sz);
    F15 = accumarray([y_idx_rite_p1, x_idx_rite   ], wxs_rite    .* wys_rite_p1 .* weight, sz);
    F16 = accumarray([y_idx_rite_p1, x_idx_rite_p1], wxs_rite_p1 .* wys_rite_p1 .* weight, sz);

    F_mesh = F01+F02+F03+F04+F05+F06+F07+F08+F09+F10+F11+F12+F13+F14+F15+F16;

end

