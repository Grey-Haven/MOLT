function ddx_inverse = get_ddx_L_x_inverse_dir(operand, x, y, dx, dy, dt, c, beta)
    
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

    ddx_inverse = zeros(N_y,N_x);
    
    % Transfer the mesh data
    for i = 1:N_x
        for j = 1:N_y
            operand_ext(j+2,i+2) = operand(j,i);
        end
    end

    % Extend the data for the operand along x
    % Corners are not needed
    for j = 1:N_y
        operand_ext(j+2,:) = polynomial_extension(operand_ext(j+2,:));
    end
    
    %==========================================================================
    % Invert the 1-D Helmholtz operator in the x-direction
    %==========================================================================
    for j = 1:N_y

        % Get the local integrals
        % Note that function names are reversed, as we use the old convention
        Nr = length(rite_moving_op(:,i));
        Nl = length(left_moving_op(:,i));
        rite_moving_op(j,:) = linear5_L(Nr, operand_ext(j+2,:), alpha, dx);
        left_moving_op(j,:) = linear5_R(Nl, operand_ext(j+2,:), alpha, dx);

        % FC step in for x operator
        [rite_moving_op(j,:), left_moving_op(j,:)] = fast_convolution(rite_moving_op(j,:), left_moving_op(j,:), alpha, dx);
        
        % Get the A and B values for Dirichlet
        % Assumes both ends of the line use Dirichlet
        %
        % See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        % By Causley, et al. 2015
        
        I_a = 0.5*( rite_moving_op(j,  1) + left_moving_op(j,  1) );
        I_b = 0.5*( rite_moving_op(j,end) + left_moving_op(j,end) );
        
        % w_a_dir = I_a - u_along_xa(t) - ( u_along_xa(t+dt) - 2*u_along_xa(t) + u_along_xa(t-dt) )/(beta**2)
        % w_b_dir = I_b - u_along_xb(t) - ( u_along_xb(t+dt) - 2*u_along_xb(t) + u_along_xb(t-dt) )/(beta**2)

        % Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a;
        w_b_dir = I_b;
        
        A_x = -( w_a_dir - mu_x*w_b_dir )/(1 - mu_x^2);
        B_x = -( w_b_dir - mu_x*w_a_dir )/(1 - mu_x^2);
        
        % Sweep the x boundary data into the operator
        for i = 1:N_x
            ddx_inverse(j,i) = -0.5*alpha*rite_moving_op(j,i) + 0.5*alpha*left_moving_op(j,i) ...
                               - alpha*A_x*exp(-alpha*(x(i  ) - x(1))) ...
                               + alpha*B_x*exp(-alpha*(x(end) - x(i)));
        end
    end

end