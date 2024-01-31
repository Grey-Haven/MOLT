function [u,dudx,dudy] = BDF1_combined_per_advance_hybrid_FFT(u, dudx, dudy, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF,kx_deriv_1,ky_deriv_1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    u(:,:,end) = BDF2_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);

    u_next = u(1:end-1,1:end-1,end);

    [N_y,N_x] = size(u(:,:,end));
    Nx = N_x - 1; % Number of nodes in the nonperiodic section of the grid
    Ny = N_y - 1; % Number of nodes in the nonperiodic section of the grid

    u_next_fft_x = fft(u_next,N_x-1,2);
    u_next_fft_y = fft(u_next,N_y-1,1);

    dudx_fft = zeros(N_y,N_x);
    dudy_fft = zeros(N_y,N_x);

    dudx_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*u_next_fft_x,N_x-1,2);
    dudy_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*u_next_fft_y,N_y-1,1);

    dudx_fft = copy_periodic_boundaries(dudx_fft);
    dudy_fft = copy_periodic_boundaries(dudy_fft);

    dudx(:,:,end) = dudx_fft;
    dudy(:,:,end) = dudy_fft;
    
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
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_y, N_x); % time history
    tmp = zeros(N_y, N_x); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(j,i) = 2*v(j,i,end-1) - v(j,i,end-2);

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + (  1/(alpha^2) )*src_data(j,i);
        end
    end

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the new time level
    u = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);

    % Shuffle is performed outside this function
    
end