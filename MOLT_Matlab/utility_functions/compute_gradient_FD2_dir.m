function [dudx_FD2,dudy_FD2] = compute_gradient_FD2_dir(u,dx,dy)

    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.
    Ny = size(u,1);
    Nx = size(u,2);
    
    dudx_FD2 = zeros(size(u));
    dudy_FD2 = zeros(size(u));

    for i = 2:Nx-1
        i_idx_m1 = i-1;
        i_idx    = i+0;
        i_idx_p1 = i+1;
        for j = 2:Ny-1
            j_idx_m1 = j-1;
            j_idx    = j+0;
            j_idx_p1 = j+1;
    
            dudx_FD2(j_idx,i_idx) = (-u(j_idx,i_idx_m1) + u(j_idx,i_idx_p1)) / (2*dx);
            dudy_FD2(j_idx,i_idx) = (-u(j_idx_m1,i_idx) + u(j_idx_p1,i_idx)) / (2*dy);
        end
    end
end