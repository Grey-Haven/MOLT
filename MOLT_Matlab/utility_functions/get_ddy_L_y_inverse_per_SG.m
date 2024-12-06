function ddy = get_ddy_L_y_inverse_per_SG(operand, x, y, dx, dy, dt, c, beta)

% with u_xx: c = 1
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_y = exp(-alpha*( y(end) - y(1) ) );
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_y, N_x);
    left_moving_op = zeros(N_y, N_x);
    
    % Extend the data for the integrand along y
    % Corners are not needed
    % operand_ext = zeros(N_y+4, N_x+4);
    
    % Transfer the mesh data
    % for i = 1:N_x
    %     for j = 1:N_y
    %         operand_ext(j+2,i+2) = operand(j,i);
    %     end
    % end

    % Extend the data for the operand along y
    % Corners are not needed
    % for i = 1:N_x
    %     operand_ext(:,i+2) = periodic_extension(operand_ext(:,i+2));
    % end

    ddy = zeros(N_y,N_x);

    %==========================================================================
    % Invert the 1-D Helmholtz operator in the y-direction
    %==========================================================================
    for i = 1:N_x
        
        % Get the local integrals 
        % Note that function names are reversed, as we use the old convention
        % Nr = length(rite_moving_op(:,i));
        % Nl = length(left_moving_op(:,i));
        rite_moving_op(:,i) = WENO_R(operand(:,i),alpha,dy,"per");
        % linear5_R(Nr, operand_ext(:,i+2), alpha, dy);
        left_moving_op(:,i) = WENO_L(operand(:,i),alpha,dy,"per");
        % linear5_L(Nl, operand_ext(:,i+2), alpha, dy);
        
        % FC step in for y operator
        [left_moving_op(:,i),rite_moving_op(:,i)] = fast_convolution(left_moving_op(:,i),rite_moving_op(:,i), alpha, dy);

        % Add the boundary terms to this convolution integral
        % *** assuming we are periodic ***

        A_y = left_moving_op(end,i)/(1 - mu_y);
        B_y = rite_moving_op(1,  i)/(1 - mu_y);
        % 
        % 
        % 
        % % Sweep the y boundary data into the operator
        % 
        for j = 1:N_y
            left_moving_op(j,i) = left_moving_op(j,i) + A_y*exp(-alpha*(y(j) - y(1)));
            rite_moving_op(j,i) = rite_moving_op(j,i) + B_y*exp(-alpha*(y(end) - y(j)));
        end
        % 
        left_moving_op(end,i) = left_moving_op(1,i);
        rite_moving_op(1,i) = rite_moving_op(end,i);

        ddy = alpha/2*(rite_moving_op - left_moving_op);


        % for j = 1:N_y
        %     ddy(j,i) = -0.5*alpha*rite_moving_op(j,i) + 0.5*alpha*left_moving_op(j,i) ...
        %              - alpha*A_y*exp(-alpha*(y(j) - y(1))) ...
        %              + alpha*B_y*exp(-alpha*(y(end) - y(j)));
        % end
    end
end