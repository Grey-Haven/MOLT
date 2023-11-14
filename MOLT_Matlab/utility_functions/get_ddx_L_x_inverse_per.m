function ddx = get_ddx_L_x_inverse_per(operand, x, y, dx, dy, dt, c, beta)
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_x = exp(-alpha*( x(end) - x(1) ) );  
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_y, N_x);
    left_moving_op = zeros(N_y, N_x);
    
    % Extend the data for the integrand along x
    % Corners are not needed
    operand_ext = zeros(N_y+4, N_x+4);
    
    % Transfer the mesh data
    for i = 1:N_x
        for j = 1:N_y
            operand_ext(j+2,i+2) = operand(j,i);
        end
    end
    
    % Extend the data for the operand along x
    % Corners are not needed
    for j = 1:N_y
        operand_ext(j+2,:) = periodic_extension(operand_ext(j+2,:));
    end
    
    ddx = zeros(N_y, N_x);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the x-direction
    %==========================================================================
    for j = 1:N_y

        % Get the local integrals
        % Note that function names are reversed, as we use the old convention
        Nr = length(rite_moving_op(j,:));
        Nl = length(left_moving_op(j,:));
        rite_moving_op(j,:) = linear5_L(Nr, operand_ext(j+2,:), alpha, dx);
        left_moving_op(j,:) = linear5_R(Nl, operand_ext(j+2,:), alpha, dx);

        % FC step
        [rite_moving_op(j,:), left_moving_op(j,:)] = fast_convolution(rite_moving_op(j,:), left_moving_op(j,:), alpha, dx);

        % Add the boundary terms to this convolution integral
        % *** assuming we are periodic ***
        A_x = (rite_moving_op(j,end) + left_moving_op(j,end))/(2 - 2*mu_x);
        B_x = (rite_moving_op(j,  1) + left_moving_op(j,  1))/(2 - 2*mu_x);
        
        % Sweep the x boundary data into the operator
        for i = 1:N_x
            ddx(j,i) = -0.5*alpha*rite_moving_op(j,i) + 0.5*alpha*left_moving_op(j,i) ...
                     - alpha*A_x*exp(-alpha*(x(i) - x(1))) ...
                     + alpha*B_x*exp(-alpha*(x(end) - x(i)));
        end
    end
end