function laplacian_u = compute_Laplacian_FD8(u,dx,dy)

    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.
    Ny = size(u,1);
    Nx = size(u,2);
    
    ddx2u_FD8 = zeros(size(u));
    ddy2u_FD8 = zeros(size(u));

    w0 = 205/72;
    w1 = 8/5;
    w2 = 1/5;
    w3 = 8/315;
    w4 = 1/560;

    for i = 1:Nx
        i_idx_m4 = i-4;
        i_idx_m3 = i-3;
        i_idx_m2 = i-2;
        i_idx_m1 = i-1;
        i_idx    = i+0;
        i_idx_p1 = i+1;
        i_idx_p2 = i+2;
        i_idx_p3 = i+3;
        i_idx_p4 = i+4;
        i_idx_m4 = mod(i_idx_m4 - 1, Nx) + 1;
        i_idx_m3 = mod(i_idx_m3 - 1, Nx) + 1;
        i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
        i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
        i_idx    = mod(i_idx    - 1, Nx) + 1;
        i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
        i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
        i_idx_p3 = mod(i_idx_p3 - 1, Nx) + 1;
        i_idx_p4 = mod(i_idx_p4 - 1, Nx) + 1;
        for j = 1:Ny
            j_idx_m4 = j-4;
            j_idx_m3 = j-3;
            j_idx_m2 = j-2;
            j_idx_m1 = j-1;
            j_idx    = j+0;
            j_idx_p1 = j+1;
            j_idx_p2 = j+2;
            j_idx_p3 = j+3;
            j_idx_p4 = j+4;
            j_idx_m4 = mod(j_idx_m4 - 1, Ny) + 1;
            j_idx_m3 = mod(j_idx_m3 - 1, Ny) + 1;
            j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
            j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
            j_idx    = mod(j_idx    - 1, Ny) + 1;
            j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
            j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
            j_idx_p3 = mod(j_idx_p3 - 1, Ny) + 1;
            j_idx_p4 = mod(j_idx_p4 - 1, Ny) + 1;

            ddx2u_FD8(j_idx,i_idx) = (-w4*u(j_idx,i_idx_m4) + w3*u(j_idx,i_idx_m3) - w2*u(j_idx,i_idx_m2) + w1*u(j_idx,i_idx_m1) - w0*u(j_idx,i_idx) ...
                                     + w1*u(j_idx,i_idx_p1) - w2*u(j_idx,i_idx_p2) + w3*u(j_idx,i_idx_p3) - w4*u(j_idx,i_idx_p4)) / (dx^2);
            
            ddy2u_FD8(j_idx,i_idx) = (-w4*u(j_idx_m4,i_idx) + w3*u(j_idx_m3,i_idx) - w2*u(j_idx_m2,i_idx) + w1*u(j_idx_m1,i_idx) - w0*u(j_idx,i_idx) ...
                                     + w1*u(j_idx_p1,i_idx) - w2*u(j_idx_p2,i_idx) + w3*u(j_idx_p3,i_idx) - w4*u(j_idx_p4,i_idx)) / (dy^2);
        end
    end

    laplacian_u = ddx2u_FD8 + ddy2u_FD8;
end