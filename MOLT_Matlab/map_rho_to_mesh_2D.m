function rho_mesh = map_rho_to_mesh_2D(x, y, dx, dy, x1, x2, ...
                                       q_s, cell_volumes, w_s)
    %%%%%%%%%
    % Computes the charge density on the mesh using 
    % the standard single level spline maps.
    %
    % Assumes a single species is present
    %%%%%%%%%

    Nx = length(x);
    Ny = length(y);

    weight = w_s*q_s;

    rho_mesh_inner = scatter_2D_vectorized_linear(Nx-1, Ny-1, x1(:), x2(:), x', y', dx, dy, weight*ones(length(x1),1));
    % rho_mesh_inner = scatter_2D_vectorized_quadratic(Nx-1, Ny-1, x1(:), x2(:), x', y', dx, dy, weight*ones(length(x1),1));
    % rho_mesh_inner = scatter_2D_vectorized_cubic(Nx-1, Ny-1, x1(:), x2(:), x', y', dx, dy, weight*ones(length(x1),1));
    rho_mesh = zeros(Ny,Nx);
    rho_mesh(1:end-1,1:end-1) = rho_mesh_inner;
    rho_mesh = copy_periodic_boundaries(rho_mesh);

    % End of particle loop
    
    % Divide by the cell volumes to compute the number density
    % Should be careful for the multi-species case. If this function
    % is called for several species, the division occurs multiple times.
    % For this, we can either: Do the division outside of this function or
    % create a rho for each particle species and apply this function (unchanged).
    rho_mesh(:,:) = rho_mesh(:,:) ./ cell_volumes(:,:);
    
    % BCs are not periodic
end