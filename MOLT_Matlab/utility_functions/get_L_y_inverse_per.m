function inverse = get_L_y_inverse_per(operand, x, y, dx, dy, dt, c, beta)
    
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
    
    % Extend the data for the integrand along y
    % Corners are not needed
    for i = 1:N_x
        operand_ext(:,i+2) = periodic_extension(operand_ext(:,i+2));
    end
            
    % Invert the y operator and apply to the operand
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        Nr = length(rite_moving_op(:,i));
        Nl = length(left_moving_op(:,i));
        rite_moving_op(:,i) = linear5_L(Nr, operand_ext(:,i+2), alpha, dy);
        left_moving_op(:,i) = linear5_R(Nl, operand_ext(:,i+2), alpha, dy);
        
        % FC step in for y operator
        [rite_moving_op(:,i), left_moving_op(:,i)] = fast_convolution(rite_moving_op(:,i), left_moving_op(:,i), alpha, dy);
        
        % Combine the integrals into the right-moving operator
        % This gives the convolution integral
        rite_moving_op(:,i) = .5*(rite_moving_op(:,i) + left_moving_op(:,i));
        
        I_a = rite_moving_op(1,  i);
        I_b = rite_moving_op(end,i);
        
        A_y = I_b/(1-mu_y);
        B_y = I_a/(1-mu_y);
        
        % Sweep the y boundary data into the operator
        rite_moving_op(:,i) = apply_A_and_B(rite_moving_op(:,i), y, alpha, A_y, B_y);
    end

    inverse = zeros(N_y, N_x);

    % Transfer contents to the inverse array
    for i = 1:N_x
        for j = 1:N_y
            inverse(j,i) = rite_moving_op(j,i);
        end
    end
end