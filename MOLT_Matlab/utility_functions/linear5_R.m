function J_R = linear5_R(N, v_ext, gamma, dx)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the fifth order approximation to the 
    % right convolution integral using a six point global stencil
    % and linear weights.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    J_R = zeros(1,N);

    % We need gamma*dx here, so we adjust the value of gamma
    gam = gamma*dx;
    
    % Get the total number of elements in v_ext (N = N_ext - 4)
    N_ext = length(v_ext);
    
    %----------------------------------------------------------------------------------------------------
    % Compute weights for the quadrature using the precomputed expressions for the left approximation
    %
    % Note: Can precompute these at the beginning of the simulation and load them later for speed
    %----------------------------------------------------------------------------------------------------
    cr_34 = ( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) )/(6*gam^3);
    cr_33 = -( 6 - 8*gam + 3*gam^2 - ( 6 - 2*gam - 2*gam^2 )*exp(-gam) )/(2*gam^3);
    cr_32 = ( 6 - 10*gam + 6*gam^2 - ( 6 - 4*gam - gam^2 + 2*gam^3 )*exp(-gam) )/(2*gam^3);
    cr_31 = -( 6 - 12*gam + 11*gam^2 - 6*gam^3 - ( 6 - 6*gam + 2*gam^2)*exp(-gam) )/(6*gam^3);
    cr_24 = ( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) )/(6*gam^3);
    cr_23 = -( 6 - 2*gam - 2*gam^2 - ( 6 + 4*gam - gam^2 - 2*gam^3 )*exp(-gam) )/(2*gam^3);
    cr_22 = ( 6 - 4*gam - gam^2 + 2*gam^3 - ( 6 + 2*gam - 2*gam^2 )*exp(-gam) )/(2*gam^3);
    cr_21 = -( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) )/(6*gam^3);
    cr_14 = ( 6 + 6*gam +2*gam^2 - ( 6 + 12*gam + 11*gam^2 + 6*gam^3 )*exp(-gam) )/(6*gam^3);
    cr_13 = -( 6 + 4*gam - gam^2 - 2*gam^3 - ( 6 + 10*gam + 6*gam^2 )*exp(-gam) )/(2*gam^3 );
    cr_12 = ( 6 + 2*gam - 2*gam^2 - ( 6 + 8*gam + 3*gam^2 )*exp(-gam) )/(2*gam^3 );
    cr_11 = -( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) )/(6*gam^3);
    
    %----------------------------------------------------------------------------------------------------
    % Compute the linear WENO weights
    %
    % Note: Can precompute these at the beginning of the simulation and load them later for speed
    %----------------------------------------------------------------------------------------------------
    d3 = ( 60 - 15*gam^2 + 2*gam^4 - ( 60 + 60*gam + 15*gam^2 - 5*gam^3 - 3*gam^4)*exp(-gam) );
    d3 = d3/(10*(gam^2)*( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) ) );
    
    d1 = ( 60 - 60*gam + 15*gam^2 + 5*gam^3 - 3*gam^4 - ( 60 - 15*gam^2 + 2*gam^4)*exp(-gam) );
    d1 = d1/(10*(gam^2)*( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) ) );
    
    d2 = 1 - d1 - d3;
        
    %----------------------------------------------------------------------------------------------------
    % Compute the local integrals J_{i}^{R} on x_{i} to x_{i+1}, i = 0,...,N
    %----------------------------------------------------------------------------------------------------
    
    % Loop through the interior points
    % Offset is from the right end-point being excluded
    for i = 3:N_ext-3
        
        % Polynomial interpolants on the smaller stencils
        p1 = cr_11*v_ext(i-2) + cr_12*v_ext(i-1) + cr_13*v_ext(i  ) + cr_14*v_ext(i+1);

        p2 = cr_21*v_ext(i-1) + cr_22*v_ext(i  ) + cr_23*v_ext(i+1) + cr_24*v_ext(i+2);

        p3 = cr_31*v_ext(i  ) + cr_32*v_ext(i+1) + cr_33*v_ext(i+2) + cr_34*v_ext(i+3);
        
        % Compute the integral using the nonlinear weights and the local polynomials
        J_R(i-2) = d1*p1 + d2*p2 + d3*p3;
    end
    J_R(end) = 0.0;
end