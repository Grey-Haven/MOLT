function [] = BDF1_combined_per_advance(u, dudx, dudy, src_data, ...
                                        x, y, t_n, dx, dy, dt, c, beta_BDF)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    BDF1_ddx_advance_per(dudx, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_ddy_advance_per(dudy, u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
    BDF1_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    
end

function [] = BDF1_ddx_advance_per(ddx, v, src_data, x, y, t, ...
                                   dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddx of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_x, N_y); % time history
    tmp = zeros(N_x, N_y); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(i,j) = 2*v(2,i,j) - v(1,i,j);

            % Contribution from the source term (at t_{n+1})
            R(i,j) = R(i,j) + ( 1/(alpha^2) )*src_data(i,j);
        end
    end
    % Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    % Shuffle is performed outside this function
end

function [] = BDF1_ddy_advance_per(ddy, v, src_data, x, y, t, ...
                                   dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddy of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_fcn.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_x, N_y); % time history
    tmp = zeros(N_x, N_y); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(i,j) = 2*v(2,i,j) - v(1,i,j);

            % Contribution from the source term (at t_{n+1})
            R(i,j) = R(i,j) + ( 1/(alpha^2) )*src_data(i,j);
        end
    end
            
    % Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    % Shuffle is performed outside this function
    
end

function [] = BDF1_advance_per(v, src_data, x, y, t, ...
                               dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates a solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_x, N_y); % time history
    tmp = zeros(N_x, N_y); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(i,j) = 2*v(2,i,j) - v(1,i,j);

            % Contribution from the source term (at t_{n+1})
            R(i,j) = R(i,j) + ( 1/(alpha^2) )*src_data(i,j);
        end
    end

    % Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_per(v(3,:,:), tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
    
end
    