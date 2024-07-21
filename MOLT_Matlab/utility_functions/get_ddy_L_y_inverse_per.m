function ddy = get_ddy_L_y_inverse_per(operand, x, y, dx, dy, dt, c, beta)
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_y = exp(-alpha*( y(end) - y(1) ) );
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_y, N_x);
    left_moving_op = zeros(N_y, N_x);
    
    % Extend the data for the integrand along y
    % Corners are not needed
    operand_ext = zeros(N_y+4, N_x+4);
    
    % Transfer the mesh data
    for i = 1:N_x
        for j = 1:N_y
            operand_ext(j+2,i+2) = operand(j,i);
        end
    end

    % Extend the data for the operand along y
    % Corners are not needed
    for i = 1:N_x
        operand_ext(:,i+2) = periodic_extension(operand_ext(:,i+2));
    end

    ddy = zeros(N_y,N_x);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the y-direction
    %==========================================================================
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        Nr = length(rite_moving_op(:,i));
        Nl = length(left_moving_op(:,i));
        rite_moving_op(:,i) = linear5_L(Nr, operand_ext(:,i+2), alpha, dy);
        left_moving_op(:,i) = linear5_R(Nl, operand_ext(:,i+2), alpha, dy);
        
        % FC step in for y operator
        [rite_moving_op(:,i), left_moving_op(:,i)] = fast_convolution(rite_moving_op(:,i), left_moving_op(:,i), alpha, dy);

        % Add the boundary terms to this convolution integral
        % *** assuming we are periodic ***
        A_y = (rite_moving_op(end,i) + left_moving_op(end,i))/(2 - 2*mu_y);
        B_y = (rite_moving_op(1,  i) + left_moving_op(1,  i))/(2 - 2*mu_y);
        
        % Sweep the y boundary data into the operator
        for j = 1:N_y
            ddy(j,i) = -0.5*alpha*rite_moving_op(j,i) + 0.5*alpha*left_moving_op(j,i) ...
                     - alpha*A_y*exp(-alpha*(y(j) - y(1))) ...
                     + alpha*B_y*exp(-alpha*(y(end) - y(j)));
        end
    end
end