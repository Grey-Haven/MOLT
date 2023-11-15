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
    F_mesh(1,:) = F_mesh(1,:) + F_mesh(end,:);
    F_mesh(:,1) = F_mesh(:,1) + F_mesh(:,end);

    F_mesh(end,:) = F_mesh(1,:);
    F_mesh(:,end) = F_mesh(:,1);

    F_mesh_update = F_mesh(:,:);
end