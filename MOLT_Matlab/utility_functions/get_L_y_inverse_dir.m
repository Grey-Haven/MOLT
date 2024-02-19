function inverse = get_L_y_inverse_dir(inverse, operand, x, y, dx, dy, dt, c, beta)
    
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

    inverse = zeros(N_y,N_x);
    
    % Transfer the mesh data
    for i = 1:N_x
        for j = 1:N_y
            operand_ext(j+2,i+2) = operand(j,i);
        end
    end
    
    % Extend the data for the integrand along y
    % Corners are not needed
    for i = 1:N_x
        operand_ext(:,i+2) = polynomial_extension(operand_ext(:,i+2));
    end
            
    % Invert the y operator and apply to the operand
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op(:,i), operand_ext(:,i+2), alpha, dy);
        linear5_R(left_moving_op(:,i), operand_ext(:,i+2), alpha, dy);
        
        % FC step in for y operator
        [rite_moving_op(:,i), left_moving_op(:,i)] = fast_convolution(rite_moving_op(:,i), left_moving_op(:,i), alpha, dy);

        % Combine the integrals into the right-moving operator
        % This gives the convolution integral
        rite_moving_op = .5*(left_moving_op(:,i) + rite_moving_op(:,i));
        
        I_a = rite_moving_op(1  ,i);
        I_b = rite_moving_op(end,i);
        
        % Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a;
        w_b_dir = I_b;
        
        A_y = -( w_a_dir - mu_y*w_b_dir )/(1 - mu_y^2);
        B_y = -( w_b_dir - mu_y*w_a_dir )/(1 - mu_y^2);
        
        % Sweep the y boundary data into the operator
        inverse(:,i) = apply_A_and_B(rite_moving_op(:,i), y, alpha, A_y, B_y);
    end
    
end
