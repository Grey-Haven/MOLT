function rho_mesh = map_rho_to_mesh_1D(x, dx, x1, q_s, cell_volumes, w_s)
    %%%%%%%%%
    % Computes the charge density on the mesh using 
    % the standard single level spline maps.
    %
    % Assumes a single species is present
    %%%%%%%%%
    
    Nx = length(x);
    
    weight = w_s*q_s;

    rho_mesh = scatter_1D_vectorized(Nx, x1, x', dx, weight);
        
    % End of particle loop
    
    % Divide by the cell volumes to compute the number density
    % Should be careful for the multi-species case. If this function
    % is called for several species, the division occurs multiple times.
    % For this, we can either: Do the division outside of this function or
    % create a rho for each particle species and apply this function (unchanged).
    rho_mesh(:) = rho_mesh(:) ./ cell_volumes(:);
end