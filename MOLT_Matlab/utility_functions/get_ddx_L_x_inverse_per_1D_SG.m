function ddx = get_ddx_L_x_inverse_per_1D_SG(operand, x, dx, dt, c, beta)
    
    N_x = length(x);
    
    alpha = beta/(c*dt);
    mu_x = exp(-alpha*( x(end) - x(1) ) );
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_x, 1);
    left_moving_op = zeros(N_x, 1);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the x-direction
    %==========================================================================


    % Get the local integrals
    % Note that function names are reversed, as we use the old convention
    rite_moving_op(:) = WENO_R(operand(:),alpha,dx,"per");
    % linear5_R(Nr, operand_ext(:,i+2), alpha, dy);
    left_moving_op(:) = WENO_L(operand(:),alpha,dx,"per");

    % FC step
    [left_moving_op, rite_moving_op] = fast_convolution(left_moving_op,rite_moving_op, alpha, dx);

    % Add the boundary terms to this convolution integral
    % *** assuming we are periodic ***
    A_x = left_moving_op(end)/(1 - mu_x);
    B_x = rite_moving_op(  1)/(1 - mu_x);
    
    % Sweep the x boundary data into the operator
    for i = 1:N_x
        left_moving_op(i) = left_moving_op(i) + A_x*exp(-alpha*(x(i) - x(1)));
        rite_moving_op(i) = rite_moving_op(i) + B_x*exp(-alpha*(x(end) - x(i)));
    end

    left_moving_op(end) = left_moving_op(1);
    rite_moving_op(1) = rite_moving_op(end);

    ddx = alpha/2*(rite_moving_op - left_moving_op);
end