function ddy = get_ddy_L_y_inverse_per(operand, x, y, dx, dy, dt, c, beta)
    
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
    
    for i = 1:N_x
        periodic_extension(operand_ext(i+2,:));
    end

    ddy = zeros(N_x,N_y);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the y-direction
    %==========================================================================
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op(i,:), operand_ext(i+2,:), alpha, dy);
        linear5_R(left_moving_op(i,:), operand_ext(i+2,:), alpha, dy);
        
        % FC step in for y operator
        fast_convolution(rite_moving_op(i,:), left_moving_op(i,:), alpha, dy);

        % Add the boundary terms to this convolution integral
        % *** assuming we are periodic ***
        A_y = (rite_moving_op(i,end) + left_moving_op(i,end))/(2 - 2*mu_y);
        B_y = (rite_moving_op(i,1) + left_moving_op(i,1))/(2 - 2*mu_y);
        
        for j = 1:N_y
            ddy(i,j) = -0.5*alpha*rite_moving_op(i,j) + 0.5*alpha*left_moving_op(i,j) ...
                     - alpha*A_y*exp(-alpha*(y(j) - y(1))) ...
                     + alpha*B_y*exp(-alpha*(y(end) - y(j)));
        end
    end
end