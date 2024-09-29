function [u,dudx,dudy] = BDF2_combined_per_advance(u, dudx, dudy, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dudx(:,:,end) = BDF2_ddx_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);

    dudy(:,:,end) = BDF2_ddy_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);
    
    u(:,:,end) = BDF2_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);
    
end

function ddx = BDF2_ddx_advance_per(v, src_data, x, y, t, ...
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
    alpha = beta/(c*dt);
    
    R = 8/3*v(:,:,end-1) - 22/9*v(:,:,end-2) + 8/9*v(:,:,end-3) - 1/9*v(:,:,end-4) + 1/(alpha^2)*src_data(:,:);

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddx = get_ddx_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
end

function ddy = BDF2_ddy_advance_per(v, src_data, x, y, t, ...
                                    dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddy of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_fcn.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    alpha = beta/(c*dt);
    
    R = 8/3*v(:,:,end-1) - 22/9*v(:,:,end-2) + 8/9*v(:,:,end-3) - 1/9*v(:,:,end-4) + 1/(alpha^2)*src_data(:,:);
            
    % Invert the y operator and apply to R, storing in tmp
    tmp = get_ddy_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddy = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
    
end

function u = BDF2_advance_per(v, src_data, x, y, t, ...
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
    alpha = beta/(c*dt);

    R = 8/3*v(:,:,end-1) - 22/9*v(:,:,end-2) + 8/9*v(:,:,end-3) - 1/9*v(:,:,end-4) + 1/(alpha^2)*src_data(:,:);

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the new time level
    u = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);

    % Shuffle is performed outside this function
    
end
    