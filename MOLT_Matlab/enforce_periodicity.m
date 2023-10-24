function [] = enforce_periodicity(F_mesh)
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Enforces periodicity in the mesh quantity.
    %
    % Helper function for the scattering step in the particle to mesh
    % mapping for the grid. In the periodic case, the last row/column of the grid
    % is redundant. So, any deposit made there will need to be transfered to the 
    % corresponding "left side" of the mesh.
    %
    % For multicomponent fields, this function can be called component-wise.
    %
    % Note: Be careful to not double count quantities on the mesh!
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Retrieve grid dimensions
    N_x = size(F_mesh,1);
    N_y = size(F_mesh,2);
    
    % Transfer the charges to enforce the periodicty
    %
    % Once the transfer is complete, then the edges
    % are copies to create identical, periodice boundaries
    
    % The code below can be verified directly by calculating
    % the charge density, assuming a unform distribution
    for i = 1:N_x
        F_mesh(i,1) = F_mesh(i,1) + F_mesh(i,end);
    end

    for j = 1:N_y        
        F_mesh(1,j) = F_mesh(1,j) + F_mesh(end,j);
    end
    
    % Copy the first row/column to the final row/column to enforce periodicity
    for j = 1:N_y 
        F_mesh(end,j) = F_mesh(1,j);
    end
    
    for i = 1:N_x
        F_mesh(i,end) = F_mesh(i,1);
    end
    
end