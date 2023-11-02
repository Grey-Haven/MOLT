function [v1_s_new, v2_s_new, P1_s_new, P2_s_new] = ...
              improved_asym_euler_momentum_push_2D2P(x1_s_new, x2_s_new, ...
                                                     P1_s_old, P2_s_old, ...
                                                     v1_s_old, v2_s_old, ...
                                                     v1_s_nm1, v2_s_nm1, ... % Needed for the Taylor approx.
                                                     ddx_psi_mesh, ddy_psi_mesh, ...
                                                     A1_mesh, ddx_A1_mesh, ddy_A1_mesh, ...
                                                     A2_mesh, ddx_A2_mesh, ddy_A2_mesh, ...
                                                     x, y, dx, dy, q_s, r_s, dt)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Applies a single step of the asymmetrical Euler method to 2D-2P particle data.
    %
    % Data is passed by reference, so there is no return. The new particle position data
    % is used to map the fields to the particles. Therefore, the position update needs
    % to be done before the momentum push.
    %
    % Note: This function is specific to the expanding beam problem.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    v1_s_new = zeros(size(v1_s_old));
    v2_s_new = zeros(size(v2_s_old));
    
    P1_s_new = zeros(size(P1_s_old));
    P2_s_new = zeros(size(P2_s_old));
    
    % Number of particles of a species s
    N_s = length(x1_s_new);
        
    for i = 1:N_s
        
        % First, we need to map the fields from the mesh to the particle
        % using the gather function based on the new particle coordinates.

        % Scalar potential data
        ddx_psi_p = gather_2D(ddx_psi_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        ddy_psi_p = gather_2D(ddy_psi_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        
        % Vector potential data
        
        % A1
        A1_p = gather_2D(A1_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        ddx_A1_p = gather_2D(ddx_A1_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        ddy_A1_p = gather_2D(ddy_A1_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        
        % A2
        A2_p = gather_2D(A2_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        ddx_A2_p = gather_2D(ddx_A2_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        ddy_A2_p = gather_2D(ddy_A2_mesh, x1_s_new(i), x2_s_new(i), x, y, dx, dy);
        
        % A3 is zero for this problem (so are its derivatives)
        
        % Compute the momentum rhs terms using a Taylor approximation of v^{n+1}
        % that retains the linear terms
        v1_s_star = v1_s_old(i) + ( v1_s_old(i) - v1_s_nm1(i) );
        v2_s_star = v2_s_old(i) + ( v2_s_old(i) - v2_s_nm1(i) );
        rhs1 = -q_s*ddx_psi_p + q_s*( ddx_A1_p*v1_s_star + ddx_A2_p*v2_s_star );
        rhs2 = -q_s*ddy_psi_p + q_s*( ddy_A1_p*v1_s_star + ddy_A2_p*v2_s_star );
        
        % Compute the new momentum
        P1_s_new(i) = P1_s_old(i) + dt*rhs1;
        P2_s_new(i) = P2_s_old(i) + dt*rhs2;
        
        % Compute the new velocity using the updated momentum
        v1_s_new(i) = (1/r_s)*(P1_s_new(i) - q_s*A1_p);
        v2_s_new(i) = (1/r_s)*(P2_s_new(i) - q_s*A2_p);
        
    end
end