function rho_mesh = map_rho_to_mesh_2D(Nx, Ny, x, y, dx, dy, ...
                                       x1, x2, ...
                                       q_s, cell_volumes, w_s)
    %%%%%%%%%
    % Computes the charge density on the mesh using 
    % the standard single level spline maps.
    %
    % Assumes a single species is present
    %%%%%%%%%
    
    % Number of simulation particles
    N_part = length(x1);
    
    weight = w_s*q_s;

    rho_mesh = zeros(Nx,Ny);
        
    % Scatter particle charge data to the mesh
    for i = 1:N_part
        rho_mesh = rho_mesh + scatter_2D(Nx, Ny, x1(i), x2(i), x, y, dx, dy, weight);
    end
        
    % End of particle loop
    
    % Divide by the cell volumes to compute the number density
    % Should be careful for the multi-species case. If this function
    % is called for several species, the division occurs multiple times.
    % For this, we can either: Do the division outside of this function or
    % create a rho for each particle species and apply this function (unchanged).
    rho_mesh(:,:) = rho_mesh(:,:) ./ cell_volumes(:,:);
    
    % BCs are not periodic
end