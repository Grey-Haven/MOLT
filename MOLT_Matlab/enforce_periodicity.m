function F_mesh_update = enforce_periodicity(F_mesh)
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
        F_mesh(1,i) = F_mesh(1,i) + F_mesh(end,i);
    end

    for j = 1:N_y        
        F_mesh(j,1) = F_mesh(j,1) + F_mesh(j,end);
    end
    
    % Copy the first row/column to the final row/column to enforce periodicity
    for j = 1:N_y 
        F_mesh(j,end) = F_mesh(j,1);
    end
    
    for i = 1:N_x
        F_mesh(end,i) = F_mesh(1,i);
    end
    F_mesh_update = F_mesh(:,:);
end