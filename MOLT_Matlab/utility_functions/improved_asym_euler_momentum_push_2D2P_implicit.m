function [v1_s_new, v2_s_new, P1_s_new, P2_s_new] = ...
              improved_asym_euler_momentum_push_2D2P(x1_s_new, x2_s_new, ...
                                                     P1_s_old, P2_s_old, ...
                                                     v1_s_old, v2_s_old, ...
                                                     v1_s_nm1, v2_s_nm1, ... % Needed for the Taylor approx.
                                                     ddx_psi_mesh, ddy_psi_mesh, ...
                                                     A1_mesh, ddx_A1_mesh, ddy_A1_mesh, ...
                                                     A2_mesh, ddx_A2_mesh, ddy_A2_mesh, ...
                                                     x, y, dx, dy, q_s, r_s, kappa, dt)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Applies a single step of the asymmetrical Euler method to 2D-2P particle data.
    %
    % Data is passed by reference, so there is no return. The new particle position data
    % is used to map the fields to the particles. Therefore, the position update needs
    % to be done before the momentum push.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     ddx_psi_p = gather_2D_vectorized(ddx_psi_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
%     ddy_psi_p = gather_2D_vectorized(ddy_psi_mesh, x1_s_new, x2_s_new, x, y, dx, dy);

    % Vector potential data
    
    % A1
%     A1_p = gather_2D_vectorized(A1_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
%     ddx_A1_p = gather_2D_vectorized(ddx_A1_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
%     ddy_A1_p = gather_2D_vectorized(ddy_A1_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
    
    % A2
%     A2_p = gather_2D_vectorized(A2_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
%     ddx_A2_p = gather_2D_vectorized(ddx_A2_mesh, x1_s_new, x2_s_new, x, y, dx, dy);
%     ddy_A2_p = gather_2D_vectorized(ddy_A2_mesh, x1_s_new, x2_s_new, x, y, dx, dy);

    meshes = zeros([size(A1_mesh),8]);
    meshes(:,:,1) = ddx_psi_mesh;
    meshes(:,:,2) = ddy_psi_mesh; 
    meshes(:,:,3) = A1_mesh;
    meshes(:,:,4) = ddx_A1_mesh;
    meshes(:,:,5) = ddy_A1_mesh;
    meshes(:,:,6) = A2_mesh;
    meshes(:,:,7) = ddx_A2_mesh;
    meshes(:,:,8) = ddy_A2_mesh;
    X = gather_2D_vectorized_multiple(meshes, x1_s_new, x2_s_new, x, y, dx, dy);
    ddx_psi_p = X(:,1);
    ddy_psi_p = X(:,2);
    A1_p = X(:,3);
    ddx_A1_p = X(:,4);
    ddy_A1_p = X(:,5);
    A2_p = X(:,6);
    ddx_A2_p = X(:,7);
    ddy_A2_p = X(:,8);

%     assert(norm(ddx_psi_p_alt - ddx_psi_p) < eps);
%     assert(norm(ddy_psi_p_alt - ddy_psi_p) < eps);
%     assert(norm(A1_p_alt - A1_p) < eps);
%     assert(norm(ddx_A1_p_alt - ddx_A1_p) < eps);
%     assert(norm(ddy_A1_p_alt - ddy_A1_p) < eps);
%     assert(norm(A2_p_alt - A2_p) < eps);
%     assert(norm(ddx_A2_p_alt - ddx_A2_p) < eps);
%     assert(norm(ddy_A2_p_alt - ddy_A2_p) < eps);
    % A3 is zero for this problem (so are its derivatives)
    
    % Compute the momentum rhs terms using a Taylor approximation of v^{n+1}
    % that retains the linear terms
    v1_s_star = v1_s_old + ( v1_s_old - v1_s_nm1 );
    v2_s_star = v2_s_old + ( v2_s_old - v2_s_nm1 );

    rhs1 = -q_s*ddx_psi_p + q_s*( ddx_A1_p.*v1_s_star + ddx_A2_p.*v2_s_star );
    rhs2 = -q_s*ddy_psi_p + q_s*( ddy_A1_p.*v1_s_star + ddy_A2_p.*v2_s_star );
    
    % Compute the new momentum
    P1_s_new = P1_s_old + dt*rhs1;
    P2_s_new = P2_s_old + dt*rhs2;
    
    denom = sqrt((P1_s_new - q_s*A1_p).^2 + (P2_s_new - q_s*A2_p).^2 + (r_s*kappa).^2);
    % Compute the new velocity using the updated momentum
    v1_s_new = (kappa*(P1_s_new - q_s*A1_p)) ./ denom;
    v2_s_new = (kappa*(P2_s_new - q_s*A2_p)) ./ denom;

end