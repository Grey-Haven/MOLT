function laplacian_u = compute_Laplacian_FD6(u,dx,dy)

    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.
    Ny = size(u,1);
    Nx = size(u,2);
    
    ddx2u_FD6 = zeros(size(u));
    ddy2u_FD6 = zeros(size(u));

    for i = 1:Nx
        i_idx_m3 = i-3;
        i_idx_m2 = i-2;
        i_idx_m1 = i-1;
        i_idx    = i+0;
        i_idx_p1 = i+1;
        i_idx_p2 = i+2;
        i_idx_p3 = i+3;
        i_idx_m3 = mod(i_idx_m3 - 1, Nx) + 1;
        i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
        i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
        i_idx    = mod(i_idx    - 1, Nx) + 1;
        i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
        i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
        i_idx_p3 = mod(i_idx_p3 - 1, Nx) + 1;
        for j = 1:Ny
            j_idx_m3 = j-3;
            j_idx_m2 = j-2;
            j_idx_m1 = j-1;
            j_idx    = j+0;
            j_idx_p1 = j+1;
            j_idx_p2 = j+2;
            j_idx_p3 = j+3;
            j_idx_m3 = mod(j_idx_m3 - 1, Nx) + 1;
            j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
            j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
            j_idx    = mod(j_idx    - 1, Ny) + 1;
            j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
            j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
            j_idx_p3 = mod(j_idx_p3 - 1, Nx) + 1;

            ddx2u_FD6(j_idx,i_idx) = (1/90*u(j_idx,i_idx_m3) + -3/20*u(j_idx,i_idx_m2) + 3/2*u(j_idx,i_idx_m1) + -49/18*u(j_idx,i_idx) + 3/2*u(j_idx,i_idx_p1) + -3/20*u(j_idx,i_idx_p2) + 1/90*u(j_idx,i_idx_p3)) / (dx^2);
            ddy2u_FD6(j_idx,i_idx) = (1/90*u(j_idx_m3,i_idx) + -3/20*u(j_idx_m2,i_idx) + 3/2*u(j_idx_m1,i_idx) + -49/18*u(j_idx,i_idx) + 3/2*u(j_idx_p1,i_idx) + -3/20*u(j_idx_p2,i_idx) + 1/90*u(j_idx_p3,i_idx)) / (dy^2);
        end
    end

    laplacian_u = ddx2u_FD6 + ddy2u_FD6;
end