function [v1_s_new, P1_s_new] = ...
              improved_asym_euler_momentum_push_1D1P(x1_s_new, ...
                                                     P1_s_old, ...
                                                     v1_s_old, ...
                                                     v1_s_nm1, ... % Needed for the Taylor approx.
                                                     ddx_psi_mesh, ...
                                                     A1_mesh, ddx_A1_mesh, ...
                                                     x, dx, q_s, r_s, kappa, dt)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Applies a single step of the asymmetrical Euler method to 2D-2P particle data.
    %
    % Data is passed by reference, so there is no return. The new particle position data
    % is used to map the fields to the particles. Therefore, the position update needs
    % to be done before the momentum push.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    meshes = zeros(length(A1_mesh),3);
    meshes(:,1) = ddx_psi_mesh;
    meshes(:,2) = A1_mesh;
    meshes(:,3) = ddx_A1_mesh;
    X = gather_1D_vectorized_multiple(meshes, x1_s_new, x, dx);
    ddx_psi_p = X(:,1);
    A1_p = X(:,2);
    ddx_A1_p = X(:,3);
    
    % Compute the momentum rhs terms using a Taylor approximation of v^{n+1}
    % that retains the linear terms
    v1_s_star = v1_s_old + ( v1_s_old - v1_s_nm1 );

    rhs1 = -q_s*ddx_psi_p + q_s*( ddx_A1_p.*v1_s_star );
    
    % Compute the new momentum
    P1_s_new = P1_s_old + dt*rhs1;
    
    denom = sqrt((P1_s_new - q_s*A1_p).^2 + (r_s*kappa).^2);
    % Compute the new velocity using the updated momentum
    v1_s_new = (kappa*(P1_s_new - q_s*A1_p)) ./ denom;

end