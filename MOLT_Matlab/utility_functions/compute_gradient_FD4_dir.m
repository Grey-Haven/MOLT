function [dudx_FD4,dudy_FD4] = compute_gradient_FD4_dir(u,dx,dy)

    % u is passed in as the domain proper. We first need to extend the
    % boundaries by extrapolating to ghost points so we can compute all the
    % derivatives.
    Ny = size(u,1);
    Nx = size(u,2);
    
    dudx_FD4 = zeros(size(u));
    dudy_FD4 = zeros(size(u));

    u_ext = zeros(Ny+4, Nx+4);
    
    % Transfer the mesh data
    u_ext(3:end-2, 3:end-2) = u;

    % Extend the data for the operand along x
    for j = 1:Ny
        u_ext(j+2,:) = polynomial_extension(u_ext(j+2,:));
    end
    % Extend the data for the operand along y
    for i = 1:Nx
        u_ext(:,i+2) = polynomial_extension(u_ext(:,i+2));
    end

    for i = 3:Nx+2
        i_idx_m2 = i-2;
        i_idx_m1 = i-1;
        i_idx    = i+0;
        i_idx_p1 = i+1;
        i_idx_p2 = i+2;
        for j = 3:Ny+2
            j_idx_m2 = j-2;
            j_idx_m1 = j-1;
            j_idx    = j+0;
            j_idx_p1 = j+1;
            j_idx_p2 = j+2;
    
            dudx_FD4(j_idx-2,i_idx-2) = (u_ext(j_idx,i_idx_m2) - 8*u_ext(j_idx,i_idx_m1) + 8*u_ext(j_idx,i_idx_p1) - u_ext(j_idx,i_idx_p2)) / (12*dx);
            dudy_FD4(j_idx-2,i_idx-2) = (u_ext(j_idx_m2,i_idx) - 8*u_ext(j_idx_m1,i_idx) + 8*u_ext(j_idx_p1,i_idx) - u_ext(j_idx_p2,i_idx)) / (12*dx);
        end
    end
end