function dudy = compute_ddy_FD6_per(u,dy)

    Ny = size(u,1);
    Nx = size(u,2);
    
    dudy_FD6 = zeros(size(u));

    for i = 1:Nx
        i_idx    = i+0;
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

            dudy_FD6(j_idx,i_idx) = (-1/60*u(j_idx_m3,i_idx) + 3/20*u(j_idx_m2,i_idx) - 3/4*u(j_idx_m1,i_idx) + 3/4*u(j_idx_p1,i_idx) - 3/20*u(j_idx_p2,i_idx) + 1/60*u(j_idx_p3,i_idx)) / (dy);
        end
    end
    dudy = dudy_FD6;
end