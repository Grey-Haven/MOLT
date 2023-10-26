function [] = linear5_L(J_L, v_ext, gamma, dx)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the fifth order approximation to the 
    % left convolution integral using a six point global stencil
    % and linear weights.
    %%%%%%%%%%%%%%%%%%%%%%%%%


    % We need gamma*dx here, so we adjust the value of gamma
    gam = gamma*dx;
    
    % Get the total number of elements in v_ext (N = N_ext - 4)
    N_ext = length(v_ext);
    
    %----------------------------------------------------------------------------------------------------
    % Compute weights for the quadrature using the precomputed expressions for the left approximation
    %
    % Note: Can precompute these at the beginning of the simulation and load them later for speed
    %----------------------------------------------------------------------------------------------------
    cl_11 = ( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) )/(6*gam^3);
    cl_12 = -( 6 - 8*gam + 3*gam^2 - ( 6 - 2*gam - 2*gam^2 )*exp(-gam) )/(2*gam^3);
    cl_13 = ( 6 - 10*gam + 6*gam^2 - ( 6 - 4*gam - gam^2 + 2*gam^3 )*exp(-gam) )/(2*gam^3);
    cl_14 = -( 6 - 12*gam + 11*gam^2 - 6*gam^3 - ( 6 - 6*gam + 2*gam^2)*exp(-gam) )/(6*gam^3);
    cl_21 = ( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) )/(6*gam^3);
    cl_22 = -( 6 - 2*gam - 2*gam^2 - ( 6 + 4*gam - gam^2 - 2*gam^3 )*exp(-gam) )/(2*gam^3);
    cl_23 = ( 6 - 4*gam - gam^2 + 2*gam^3 - ( 6 + 2*gam - 2*gam^2 )*exp(-gam) )/(2*gam^3);
    cl_24 = -( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) )/(6*gam^3);
    cl_31 = ( 6 + 6*gam +2*gam^2 - ( 6 + 12*gam + 11*gam^2 + 6*gam^3 )*exp(-gam) )/(6*gam^3);
    cl_32 = -( 6 + 4*gam - gam^2 - 2*gam^3 - ( 6 + 10*gam + 6*gam^2 )*exp(-gam) )/(2*gam^3 );
    cl_33 = ( 6 + 2*gam - 2*gam^2 - ( 6 + 8*gam + 3*gam^2 )*exp(-gam) )/(2*gam^3 );
    cl_34 = -( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) )/(6*gam^3);
        
    %----------------------------------------------------------------------------------------------------
    % Compute the linear WENO weights
    % Note: Can precompute these at the beginning of the simulation and load them later for speed
    %----------------------------------------------------------------------------------------------------
    d1 = ( 60 - 15*gam^2 + 2*gam^4 - ( 60 + 60*gam + 15*gam^2 - 5*gam^3 - 3*gam^4)*exp(-gam) );
    d1 = d1/(10*(gam^2)*( 6 - 6*gam + 2*gam^2 - ( 6 - gam^2 )*exp(-gam) ) );

    d3 = ( 60 - 60*gam + 15*gam^2 + 5*gam^3 - 3*gam^4 - ( 60 - 15*gam^2 + 2*gam^4)*exp(-gam) ) ;
    d3 = d3/(10*(gam^2)*( 6 - gam^2 - ( 6 + 6*gam + 2*gam^2 )*exp(-gam) ) );

    d2 = 1 - d1 - d3;
        
    %----------------------------------------------------------------------------------------------------
    % Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
    %----------------------------------------------------------------------------------------------------

    J_L(1) = 0.0;
    
    % Loop through the interior points of the extended array
    % Offset is from the left end-point being excluded
    for i = 4:N_ext-2
        % Polynomial interpolants on the smaller stencils
        p1 = cl_11*v_ext(i-3) + cl_12*v_ext(i-2) + cl_13*v_ext(i-1) + cl_14*v_ext(i  );
        
        p2 = cl_21*v_ext(i-2) + cl_22*v_ext(i-1) + cl_23*v_ext(i  ) + cl_24*v_ext(i+1);
        
        p3 = cl_31*v_ext(i-1) + cl_32*v_ext(i  ) + cl_33*v_ext(i+1) + cl_34*v_ext(i+2);
        
        % Compute the integral using the nonlinear weights and the local polynomials
        J_L(i-2) = d1*p1 + d2*p2 + d3*p3;
    end
end