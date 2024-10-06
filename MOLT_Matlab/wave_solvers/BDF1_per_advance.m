function u = BDF1_per_advance(v, src_data, x, y, dx, dy, dt, c, beta)
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
    
    % Variables for the integrands
    R   = 2*v(:,:,end-1) - v(:,:,end-2) + (  1/(alpha^2) )*src_data(:,:); % time history

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the new time level
    u = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);

    % Shuffle is performed outside this function
    
end