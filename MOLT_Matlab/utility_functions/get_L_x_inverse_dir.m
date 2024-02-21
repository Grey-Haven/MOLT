function inverse = get_L_x_inverse_dir(operand, x, y, dx, dy, dt, c, beta)
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_x = exp(-alpha*( x(end) - x(1) ) );
    
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
    for j = 1:N_y
        operand_ext(j+2,:) = polynomial_extension(operand_ext(j+2,:));
    end
            
    % Invert the y operator and apply to the operand
    for j = 1:N_y
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        Nr = length(rite_moving_op(j,:));
        Nl = length(left_moving_op(j,:));
        rite_moving_op(j,:) = linear5_L(Nr, operand_ext(j+2,:), alpha, dx);
        left_moving_op(j,:) = linear5_R(Nl, operand_ext(j+2,:), alpha, dx);
        
        % FC step in for y operator
        [rite_moving_op(j,:), left_moving_op(j,:)] = fast_convolution(rite_moving_op(j,:), left_moving_op(j,:), alpha, dx);

        % Combine the integrals into the right-moving operator
        % This gives the convolution integral
        rite_moving_op(j,:) = .5*(rite_moving_op(j,:) + left_moving_op(j,:));
        
        I_a = rite_moving_op(j,   1);
        I_b = rite_moving_op(j, end);
        
        % Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a;
        w_b_dir = I_b;
        
        A_x = -( w_a_dir - mu_x*w_b_dir )/(1 - mu_x^2);
        B_x = -( w_b_dir - mu_x*w_a_dir )/(1 - mu_x^2);

        % Sweep the x boundary data into the operator
        inverse(j,:) = apply_A_and_B(rite_moving_op(j,:), x, alpha, A_x, B_x);
    end
    
end