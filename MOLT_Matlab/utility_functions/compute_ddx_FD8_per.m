function dudx = compute_ddx_FD8_per(u,dx)

    Ny = size(u,1);
    Nx = size(u,2);
    
    dudx_FD8 = zeros(size(u));

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
            j_idx = j;
            dudx_FD8(j_idx,i_idx) = (1/280*u(j_idx,i_idx_m4) - 4/105*u(j_idx,i_idx_m3) + 1/5*u(j_idx,i_idx_m2) - 4/5*u(j_idx,i_idx_m1) ...
                                   + 4/5*u(j_idx,i_idx_p1) - 1/5*u(j_idx,i_idx_p2) + 4/105*u(j_idx,i_idx_p3) - 1/280*u(j_idx,i_idx_p4)) / dx;
        end
    end
    dudx = dudx_FD8;
end