function [u,dudx] = BDF1_combined_per_advance_1D(u, dudx, src_data, x, t_n, dx, dt, c, beta_BDF)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dudx(:,end) = BDF1_ddx_advance_per_1D(u, src_data, x, t_n, dx, dt, c, beta_BDF);
    
    u(:,end) = BDF1_advance_per_1D(u, src_data, x, t_n, dx, dt, c, beta_BDF);
    
end

function ddx = BDF1_ddx_advance_per_1D(v, src_data, x, t, dx, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddx of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Build the convolution integral over the time history R
    
    alpha = beta/(c*dt);

    R = 2*v(:,end-1) - v(:,end-2) + (1/alpha^2)*src_data;
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddx = get_ddx_L_x_inverse_per_1D(R, x, dx, dt, c, beta);
    
    % Shuffle is performed outside this function
end

function u = BDF1_advance_per_1D(v, src_data, x, t, dx, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates a solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    alpha = beta/(c*dt);

    R = 2*v(:,end-1) - v(:,end-2) + (1/alpha^2)*src_data;

    % Invert the x operator and apply to tmp, then store in the new time level
    u = get_L_x_inverse_per_1D(R, x, dx, dt, c, beta);

    % Shuffle is performed outside this function
    
end
    