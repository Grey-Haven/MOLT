function F_ps = gather_2D_vectorized_multiple_cubic(F_meshes, x1_p, x2_p, x, y, dx, dy)
    %%%%%%%%
    % Gathers the field information corresponding to a single particle
    % with coordinates (x1_p, x2_p), but does so with multiple fields to
    % reduce stack traversal.
    %
    % This function uses cubic splines to map mesh data to the particle
    %
    % Unlike the linear method, this assumes a grid of 
    % [a_x, b_x) x [a_y, b_y), that is, it does not include the right and
    % top boundaries.
    %%%%%%%%

    try

        Nx = length(x)-1;
        Ny = length(y)-1;
                            
        F_ps = zeros(length(x1_p),size(F_meshes,3));
    
        index_offset = 1;
    
        % Left/Bottom-most indices
        x_idx_left = index_offset + floor((x1_p - x(1))/dx);
        y_idx_left = index_offset + floor((x2_p - y(1))/dy);
    
        x_idx_left_m1 = x_idx_left - 1;
        x_idx_rite = x_idx_left + 1;
        x_idx_rite_p1 = x_idx_rite + 1;
    
        y_idx_left_m1 = y_idx_left - 1;
        y_idx_rite = y_idx_left + 1;
        y_idx_rite_p1 = y_idx_rite + 1;
    
        % With cubic, like linear, the nodes are the four closest to the
        % particle: p
        % ie 
        %
        % N_l2        N_l1  p     N_r1        N_r2
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
    
        idx01 = sub2ind(size(F_meshes(:,:,1)), y_idx_left_m1, x_idx_left_m1);
        idx02 = sub2ind(size(F_meshes(:,:,1)), y_idx_left_m1, x_idx_left   );
        idx03 = sub2ind(size(F_meshes(:,:,1)), y_idx_left_m1, x_idx_rite   );
        idx04 = sub2ind(size(F_meshes(:,:,1)), y_idx_left_m1, x_idx_rite_p1);
    
        idx05 = sub2ind(size(F_meshes(:,:,1)), y_idx_left   , x_idx_left_m1);
        idx06 = sub2ind(size(F_meshes(:,:,1)), y_idx_left   , x_idx_left   );
        idx07 = sub2ind(size(F_meshes(:,:,1)), y_idx_left   , x_idx_rite   );
        idx08 = sub2ind(size(F_meshes(:,:,1)), y_idx_left   , x_idx_rite_p1);
    
        idx09 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite   , x_idx_left_m1);
        idx10 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite   , x_idx_left   );
        idx11 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite   , x_idx_rite   );
        idx12 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite   , x_idx_rite_p1);
    
        idx13 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite_p1, x_idx_left_m1);
        idx14 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite_p1, x_idx_left   );
        idx15 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite_p1, x_idx_rite   );
        idx16 = sub2ind(size(F_meshes(:,:,1)), y_idx_rite_p1, x_idx_rite_p1);
    
        for i = 1:size(F_meshes,3)
    
            F_mesh = F_meshes(:,:,i);
    
            F01 = F_mesh(idx01) .* wys_left_m1 .* wxs_left_m1;
            F02 = F_mesh(idx02) .* wys_left_m1 .* wxs_left   ;
            F03 = F_mesh(idx03) .* wys_left_m1 .* wxs_rite   ;
            F04 = F_mesh(idx04) .* wys_left_m1 .* wxs_rite_p1;
    
            F05 = F_mesh(idx05) .* wys_left    .* wxs_left_m1;
            F06 = F_mesh(idx06) .* wys_left    .* wxs_left   ;
            F07 = F_mesh(idx07) .* wys_left    .* wxs_rite   ;
            F08 = F_mesh(idx08) .* wys_left    .* wxs_rite_p1;
    
            F09 = F_mesh(idx09) .* wys_rite    .* wxs_left_m1;
            F10 = F_mesh(idx10) .* wys_rite    .* wxs_left   ;
            F11 = F_mesh(idx11) .* wys_rite    .* wxs_rite   ;
            F12 = F_mesh(idx12) .* wys_rite    .* wxs_rite_p1;
    
            F13 = F_mesh(idx13) .* wys_rite_p1 .* wxs_left_m1;
            F14 = F_mesh(idx14) .* wys_rite_p1 .* wxs_left   ;
            F15 = F_mesh(idx15) .* wys_rite_p1 .* wxs_rite   ;
            F16 = F_mesh(idx16) .* wys_rite_p1 .* wxs_rite_p1;
    
            F_p = F01+F02+F03+F04+F05+F06+F07+F08+F09+F10+F11+F12+F13+F14+F15+F16;
    
            F_ps(:,i) = F_p;
        end
    catch exception
        throw(exception);
    end
end