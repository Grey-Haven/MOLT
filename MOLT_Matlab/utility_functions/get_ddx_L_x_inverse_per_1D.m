function ddx = get_ddx_L_x_inverse_per_1D(operand, x, dx, dt, c, beta)
    
    N_x = length(x);
    
    alpha = beta/(c*dt);
    mu_x = exp(-alpha*( x(end) - x(1) ) );
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_x, 1);
    left_moving_op = zeros(N_x, 1);
    
    % Extend the data for the integrand along x
    % Corners are not needed
    operand_ext = zeros(N_x+4, 1);
    
    % Transfer the mesh data
    for i = 1:N_x
        operand_ext(i+2) = operand(i);
    end

    % Extend the data for the operand along x
    operand_ext = periodic_extension(operand_ext);
    
    ddx = zeros(N_x, 1);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the x-direction
    %==========================================================================

    % Get the local integrals
    % Note that function names are reversed, as we use the old convention
    Nr = length(rite_moving_op(:));
    Nl = length(left_moving_op(:));
    rite_moving_op(:) = linear5_L(Nr, operand_ext(:), alpha, dx);
    left_moving_op(:) = linear5_R(Nl, operand_ext(:), alpha, dx);

    % FC step
    [rite_moving_op, left_moving_op] = fast_convolution(rite_moving_op, left_moving_op, alpha, dx);

    % Add the boundary terms to this convolution integral
    % *** assuming we are periodic ***
    A_x = (rite_moving_op(end) + left_moving_op(end))/(2 - 2*mu_x);
    B_x = (rite_moving_op(  1) + left_moving_op(  1))/(2 - 2*mu_x);
    
    % Sweep the x boundary data into the operator
    for i = 1:N_x
        ddx(i) = -0.5*alpha*rite_moving_op(i) + 0.5*alpha*left_moving_op(i) ...
                 - alpha*A_x*exp(-alpha*(x(i) - x(1))) ...
                 + alpha*B_x*exp(-alpha*(x(end) - x(i)));
    end
end