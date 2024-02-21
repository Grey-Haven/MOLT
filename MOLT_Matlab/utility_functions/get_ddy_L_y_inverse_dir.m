function ddy_inverse = get_ddy_L_y_inverse_dir(operand, x, y, dx, dy, dt, c, beta)
    
    N_x = length(x);
    N_y = length(y);
    
    alpha = beta/(c*dt);
    mu_y = exp(-alpha*( y(end) - y(1) ) );
    
    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_y, N_x);
    left_moving_op = zeros(N_y, N_x);
    
    % Extend the data for the integrand along y
    % Corners are not needed
    operand_ext = zeros(N_y+4, N_y+4);

    ddy_inverse = zeros(N_y,N_x);
    
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

        % Get the A and B values for Dirichlet
        % Assumes both ends of the line use Dirichlet
        %
        % See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        % By Causley, et al. 2015

        I_a = 0.5*( rite_moving_op(1,i) + left_moving_op(1,i) );
        I_b = 0.5*( rite_moving_op(end,i) + left_moving_op(end,i) );
        
        % w_a_dir = I_a - u_along_xa(t) - ( u_along_xa(t+dt) - 2*u_along_xa(t) + u_along_xa(t-dt) )/(beta**2)
        % w_b_dir = I_b - u_along_xb(t) - ( u_along_xb(t+dt) - 2*u_along_xb(t) + u_along_xb(t-dt) )/(beta**2)

        % Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a;
        w_b_dir = I_b;
        
        A_y = -( w_a_dir - mu_y*w_b_dir )/(1 - mu_y^2);
        B_y = -( w_b_dir - mu_y*w_a_dir )/(1 - mu_y^2);

        % Sweep the y boundary data into the operator
        for j = 1:N_y
            ddy_inverse(j,i) = -0.5*alpha*rite_moving_op(j,i) + 0.5*alpha*left_moving_op(j,i) ...
                               - alpha*A_y*exp(-alpha*(y(j  ) - y(1))) ...
                               + alpha*B_y*exp(-alpha*(y(end) - y(j)));
        end
    end
    
end