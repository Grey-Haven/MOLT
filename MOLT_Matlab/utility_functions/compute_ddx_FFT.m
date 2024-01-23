function dudx_fft = compute_ddx_FFT(u,kx_deriv_1)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an x derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%

    [N_y,N_x] = size(u(:,:,end));

    u_fft_x = fft(u_next,N_x-1,2);

    dudx_fft = zeros(N_y,N_x);

    dudx_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*u_fft_x,N_x-1,2);

    dudx_fft = copy_periodic_boundaries(dudx_fft);
end