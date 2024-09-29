function u_fine = solve_helmholtz_MG_FD8(u_guess, RHS, alpha, dx, dy)
    
    % Assumes the right and upper boundary are within the boundary
    % That is, if the domain is [ax,bx]X[ay,by], the rightmost column of
    % nodes are the nodes to the left of bx, the upmost row of nodes are
    % the nodes below by.

    [Ny_fine,Nx_fine] = size(RHS);
    
    % We have the system Au = b
    % Here b is the RHS, A is the FD8 matrix
    % The 1D FD8 Second Derivative matrix has the form 
    % (0,0,...,0,1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90, 0, ..., 0)
    % We are solving the Helmholtz equation, which is (I - 1/alpha^2 FD8)u = RHS
    % So A = I-FD8

    % Assert the grid is a perfect square
    assert(Ny_fine == Nx_fine);
    % Assert the grid dimensions are powers of 2
    assert(Nx_fine~=0 & (bitand(Nx_fine,Nx_fine-1)==0));

    TOL = 1e-14;
    MAX = 10000;

    if (Nx_fine <= 8)
        % No discernible difference between these
        % u_fine = solve_helmholtz_FD8(RHS, alpha, dx, dy);
        u_fine = solve_helmholtz_GS_FD8(u_guess, RHS, alpha, dx, dy, MAX, TOL);
        return;
    end

    Nx_coarse = Nx_fine / 2;
    Ny_coarse = Ny_fine / 2;

    u_fine = u_guess;
    u_coarse_guess = zeros(Ny_coarse, Nx_coarse);
    res_coarse = zeros(Ny_coarse, Nx_coarse);

    alpha2 = alpha^2;

    for k = 1:MAX

        % Relaxation/Smoothing
        u_fine = solve_helmholtz_GS_FD8(u_fine, RHS, alpha, dx, dy, 100, 1e-2);

        % Compute residual
        laplacian_u_num = compute_Laplacian_FD8(u_fine, dx, dy);
        Au_num = (u_fine - 1/alpha2*laplacian_u_num);
        res_fine = RHS - Au_num;

        % Restriction
        for i_H = 1:Nx_coarse
            for j_H = 1:Ny_coarse

                i_h = 2*i_H;
                j_h = 2*j_H;

                i_hm1 = mod(i_h - 2, Nx_fine) + 1;
                i_hp1 = mod(i_h + 0, Nx_fine) + 1;
                j_hm1 = mod(j_h - 2, Ny_fine) + 1;
                j_hp1 = mod(j_h + 0, Ny_fine) + 1;
                
                corners = res_fine(j_hm1, i_hm1) + res_fine(j_hp1, i_hm1) + res_fine(j_hm1, i_hp1) + res_fine(j_hp1, i_hp1);
                edges = res_fine(j_h, i_hm1) + res_fine(j_h, i_hp1) + res_fine(j_hm1, i_h) + res_fine(j_hp1, i_h);
                center = res_fine(j_h, i_h);

                res_coarse(j_H, i_H) = 1/16*corners + 1/8*edges + 1/4*center;
                % res_coarse(j_H, i_H) = center;
            end
        end

        % Coarse Grid Correction
        % e_H = solve_helmholtz_GS_FD8(u_coarse_guess, res_coarse, alpha, dx, dy, MAX, TOL);
        e_H = solve_helmholtz_MG_FD8(u_coarse_guess, res_coarse, alpha, dx, dy);
        % Adds a W cycle
        e_H = solve_helmholtz_MG_FD8(e_H, res_coarse, alpha, dx, dy);

        % Interpolation/Prolongation
        e_h = prolong(e_H);

        % Post-Smoothing
        u_fine = u_fine + e_h;

        if (max(max(abs(res_fine))) < TOL)
            break;
        end
    end

    if (k == MAX)
        ME = MException('HelmholtzSolverException','Solver was unable to reach satisfactory TOL');
        throw(ME);
    end
end

function e_h = prolong(e_H)
    
    [Ny_coarse,Nx_coarse] = size(e_H);
    Nx_fine = 2*Nx_coarse;
    Ny_fine = 2*Ny_coarse;

    e_h = zeros(Ny_fine, Nx_fine);

    % Copy coarse grid points directly to fine grid
    e_h(1:2:end, 1:2:end) = e_H;

    % Interpolate in the x-direction
    e_h(1:2:end, 2:2:end-1) = 0.5 * (e_H(:, 1:end-1) + e_H(:, 2:end));

    % Interpolate in the y-direction
    e_h(2:2:end-1, 1:2:end) = 0.5 * (e_H(1:end-1, :) + e_H(2:end, :));

    % Interpolate in both x- and y-directions
    e_h(2:2:end-1, 2:2:end-1) = 0.25 * (e_H(1:end-1, 1:end-1) + e_H(2:end, 1:end-1) ...
                                      + e_H(1:end-1, 2:end  ) + e_H(2:end, 2:end  ));

    e_h(end, 1:2:end-1) =  0.5 * (e_H(end-1, :) + e_H(1, :));
    e_h(1:2:end-1,end ) =  0.5 * (e_H(:, end-1) + e_H(:, 1));

    % Interpolate last column
    e_h(2:2:end-2,end) =  0.25 * (e_H(1:end-1, end-1) + e_H(2:end, end-1) + e_H(1:end-1, 1) + e_H(2:end, 1));

    % Interpolate last row
    e_h(end,2:2:end-2) =  0.25 * (e_H(end-1, 1:end-1) + e_H(end-1, 2:end) + e_H(1, 1:end-1) + e_H(1, 2:end));

    % Interpolate last element
    e_h(end,end) = 0.25*(e_H(end, end) + e_H(end, 1) + e_H(1,end) + e_H(1, 1));
end