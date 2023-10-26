function [] = get_L_y_inverse_per(inverse, operand, x, y, dx, dy, dt, c, beta)
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_y = exp(-alpha*( y(end) - y(1) ) );    
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_x, N_y);
    left_moving_op = zeros(N_x, N_y);
    
    % Extend the data for the integrand along y
    % Corners are not needed
    operand_ext = zeros(N_x+4, N_y+4);
    
    % Transfer the mesh data
    for i = 1:N_x
        for j = 1:N_y
            operand_ext(i+2,j+2) = operand(i,j);
        end
    end
    
    % Extend the data for the integrand along y
    % Corners are not needed
    for i = 1:N_x
        periodic_extension(operand_ext(i+2,:));
    end
            
    % Invert the y operator and apply to the operand
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op(i,:), operand_ext(i+2,:), alpha, dy)
        linear5_R(left_moving_op(i,:), operand_ext(i+2,:), alpha, dy)
        
        % FC step in for y operator
        fast_convolution(rite_moving_op(i,:), left_moving_op(i,:), alpha, dy)
        
        % Combine the integrals into the right-moving operator
        % This gives the convolution integral
        rite_moving_op(i,:) = .5*(rite_moving_op(i,:) + left_moving_op(i,:));
        
        I_a = rite_moving_op(i,1);
        I_b = rite_moving_op(i,end);
        
        A_y = I_b/(1-mu_y);
        B_y = I_a/(1-mu_y);
        
        % Sweep the y boundary data into the operator
        apply_A_and_B(rite_moving_op(i,:), y, alpha, A_y, B_y);
    end

    % Transfer contents to the inverse array
    for i = 1:N_x
        for j = 1:N_y
            inverse(i,j) = rite_moving_op(i,j);
        end
    end
end