function [dudx,dudy] = compute_derivative_FD6_periodic(u,dx,dy)

    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.
    Ny = size(u,1);
    Nx = size(u,2);
    
    dudx_FD6 = zeros(size(u)+1);
    dudy_FD6 = zeros(size(u)+1);

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

            dudx_FD6(j_idx,i_idx) = (-1/60*u(j_idx,i_idx_m3) + 3/20*u(j_idx,i_idx_m2) - 3/4*u(j_idx,i_idx_m1) + 3/4*u(j_idx,i_idx_p1) - 3/20*u(j_idx,i_idx_p2) + 1/60*u(j_idx,i_idx_p3)) / (dx);
            dudy_FD6(j_idx,i_idx) = (-1/60*u(j_idx_m3,i_idx) + 3/20*u(j_idx_m2,i_idx) - 3/4*u(j_idx_m1,i_idx) + 3/4*u(j_idx_p1,i_idx) - 3/20*u(j_idx_p2,i_idx) + 1/60*u(j_idx_p3,i_idx)) / (dy);
        end
    end

    dudx_FD6 = copy_periodic_boundaries(dudx_FD6);
    dudy_FD6 = copy_periodic_boundaries(dudy_FD6);

    dudx = dudx_FD6;
    dudy = dudy_FD6;
end