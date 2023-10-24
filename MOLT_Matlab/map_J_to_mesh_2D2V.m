function [] = map_J_to_mesh_2D2V(J_mesh, x, y, dx, dy, ...
                       x1, x2, v1, v2, ...
                       q_s, cell_volumes, w_s)
    %%%%%%%%%%%%%%%%%%%%%
    % Computes the current density for the field solvers using velocity information
    % in the 2D-2V setting.
    %
    % This mapping is to be used for the expanding beam problem.
    %%%%%%%%%%%%%%%%%%%%%
    
    % Number of simulation particles
    N_part = length(x1);
    
    weight = w_s*q_s;
    
    J1_mesh = squeeze(J_mesh(1,:,:));
    J2_mesh = squeeze(J_mesh(2,:,:));

    % Scatter current to the mesh
    for i = 1:N_part
        
        weight1 = weight*v1(i);
        weight2 = weight*v2(i);
        
        scatter_2D(J1_mesh, x1(i), x2(i), x, y, dx, dy, weight1); % J_x
        scatter_2D(J2_mesh, x1(i), x2(i), x, y, dx, dy, weight2); % J_y
       
    % End of particle loop
    
    % Divide by the cell volumes to compute the number density
    % Should be careful for the multi-species case. If this function
    % is called for several species, the division occurs multiple times.
    % For this, we can either: Do the division outside of this function or
    % create a rho for each particle species and apply this function (unchanged).
    J_mesh(1,:,:) = J1_mesh(:,:) ./ cell_volumes(:,:);
    J_mesh(2,:,:) = J2_mesh(:,:) ./ cell_volumes(:,:);
    
    % BCs are not periodic
    end
end