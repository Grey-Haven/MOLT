function [u,dudx,dudy] = BDF2_combined_per_advance_hybrid_FFT(u, dudx, dudy, src_data, x, y, dx, dy, dt, c, beta_BDF, kx_deriv_1, ky_deriv_1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    u(:,:,end) = BDF2_per_advance(u, src_data, x, y, dx, dy, dt, c, beta_BDF);

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