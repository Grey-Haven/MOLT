function v_next = BDF2_advance_dir(v, src_data, x, y, dx, dy, dt, c, beta)
    %%%
    % Calculates a solution to the wave equation using the second-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%
    % Build the convolution integral over the time history R
    alpha = beta/(c*dt);

    % R = 8/3*v(:,:,end-1) - 22/9*v(:,:,end-2) + 8/9*v(:,:,end-3) - 1/9*v(:,:,end-4) + 1/(alpha^2)*src_data;
    % R = 0.5*( 5*v(:,:,end-1) - 4*v(:,:,end-2) + v(:,:,end-3) ) + ( 1/(alpha^2) )*src_data(:,:);
    R = 2*v(:,:,end-1) - v(:,:,end-2) + (1/(alpha^2))*src_data(:,:);

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_dir(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the new time level
    v_next = get_L_x_inverse_dir(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
end