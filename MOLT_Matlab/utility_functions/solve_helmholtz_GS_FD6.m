function u_GS = solve_helmholtz_GS_FD6(u_GS, RHS, alpha, dx, dy, MAX, TOL)
    
    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.

    [N_y,N_x] = size(RHS);
    
    % b = RHS(1:end-1,1:end-1);
    
    % We have the system Au = b
    % Here b is the RHS, A is the FD6 matrix
    % The 1D FD6 Second Derivative matrix has the form 
    % (0,0,...,0,1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90, 0, ..., 0)
    % We are solving the Helmholtz equation, which is (I - 1/alpha^2 FD6)u = RHS
    % So A = I-FD6

    alpha2_inv = 1/alpha^2;

    dx2 = dx^2;
    dy2 = dy^2;

    a_ij   = 1 - alpha2_inv*(-(49/18) / dx2 + -(49/18) / dy2);

    a_ij_i_m3 = 0 - alpha2_inv*(1/90) / dx2;
    a_ij_i_m2 = 0 - alpha2_inv*(-3/20) / dx2;
    a_ij_i_m1 = 0 - alpha2_inv*(3/2) / dx2;

    a_ij_i_p1 = 0 - alpha2_inv*(3/2) / dx2;
    a_ij_i_p2 = 0 - alpha2_inv*(-3/20) / dx2;
    a_ij_i_p3 = 0 - alpha2_inv*(1/90) / dx2;

    a_ij_j_m3 = 0 - alpha2_inv*(1/90) / dy2;
    a_ij_j_m2 = 0 - alpha2_inv*(-3/20) / dy2;
    a_ij_j_m1 = 0 - alpha2_inv*(3/2) / dy2;

    a_ij_j_p1 = 0 - alpha2_inv*(3/2) / dy2;
    a_ij_j_p2 = 0 - alpha2_inv*(-3/20) / dy2;
    a_ij_j_p3 = 0 - alpha2_inv*(1/90) / dy2;

    % Nx = N_x - 1;
    % Ny = N_y - 1;

    for k = 1:MAX

        % max_diff = 0;

        u_prev = u_GS;

        for i = 1:N_x
            i_idx_m3 = i-3;
            i_idx_m2 = i-2;
            i_idx_m1 = i-1;
            i_idx    = i+0;
            i_idx_p1 = i+1;
            i_idx_p2 = i+2;
            i_idx_p3 = i+3;
            i_idx_m3 = mod(i_idx_m3 - 1, N_x) + 1;
            i_idx_m2 = mod(i_idx_m2 - 1, N_x) + 1;
            i_idx_m1 = mod(i_idx_m1 - 1, N_x) + 1;
            i_idx    = mod(i_idx    - 1, N_x) + 1;
            i_idx_p1 = mod(i_idx_p1 - 1, N_x) + 1;
            i_idx_p2 = mod(i_idx_p2 - 1, N_x) + 1;
            i_idx_p3 = mod(i_idx_p3 - 1, N_x) + 1;
            for j = 1:N_y

                j_idx_m3 = j-3;
                j_idx_m2 = j-2;
                j_idx_m1 = j-1;
                j_idx    = j+0;
                j_idx_p1 = j+1;
                j_idx_p2 = j+2;
                j_idx_p3 = j+3;
                j_idx_m3 = mod(j_idx_m3 - 1, N_x) + 1;
                j_idx_m2 = mod(j_idx_m2 - 1, N_y) + 1;
                j_idx_m1 = mod(j_idx_m1 - 1, N_y) + 1;
                j_idx    = mod(j_idx    - 1, N_y) + 1;
                j_idx_p1 = mod(j_idx_p1 - 1, N_y) + 1;
                j_idx_p2 = mod(j_idx_p2 - 1, N_y) + 1;
                j_idx_p3 = mod(j_idx_p3 - 1, N_x) + 1;

                b_ij = RHS(j_idx, i_idx);

                x_ij_i_m3 = u_GS(j_idx, i_idx_m3);
                x_ij_i_m2 = u_GS(j_idx, i_idx_m2);
                x_ij_i_m1 = u_GS(j_idx, i_idx_m1);
                % x_ij     = x_GS(j_idx, i_idx   );
                x_ij_i_p1 = u_GS(j_idx, i_idx_p1);
                x_ij_i_p2 = u_GS(j_idx, i_idx_p2);
                x_ij_i_p3 = u_GS(j_idx, i_idx_p3);

                x_ij_j_m3 = u_GS(j_idx_m3, i_idx);
                x_ij_j_m2 = u_GS(j_idx_m2, i_idx);
                x_ij_j_m1 = u_GS(j_idx_m1, i_idx);
                % x_ij     = x_GS(j_idx   , i_idx);
                x_ij_j_p1 = u_GS(j_idx_p1, i_idx);
                x_ij_j_p2 = u_GS(j_idx_p2, i_idx);
                x_ij_j_p3 = u_GS(j_idx_p3, i_idx);

                u_GS(j_idx, i_idx) = 1/a_ij * (b_ij ...
                                             - a_ij_i_m3*x_ij_i_m3 - a_ij_i_m2*x_ij_i_m2 - a_ij_i_m1*x_ij_i_m1 ...
                                             - a_ij_j_m3*x_ij_j_m3 - a_ij_j_m2*x_ij_j_m2 - a_ij_j_m1*x_ij_j_m1 ...
                                             - a_ij_i_p1*x_ij_i_p1 - a_ij_i_p2*x_ij_i_p2 - a_ij_i_p3*x_ij_i_p3 ...
                                             - a_ij_j_p1*x_ij_j_p1 - a_ij_j_p2*x_ij_j_p2 - a_ij_j_p3*x_ij_j_p3);
            end
        end

        if (max(max(abs(u_prev - u_GS))) < TOL)
            break;
        end
    end

    if (k == MAX)
        ME = MException('HelmholtzSolverException','Solver was unable to reach satisfactory TOL');
        throw(ME);
    end

    % u = copy_periodic_boundaries(u_GS);
end