function dudy_fft = compute_ddy_FFT(u,ky_deriv_1)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an x derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%

    [N_y,N_x] = size(u(:,:,end));
    
    u_domain = u(1:end-1,1:end-1,end);

    u_fft_y = fft(u_domain,N_y-1,1);

    dudy_fft = zeros(N_y,N_x);

    dudy_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*u_fft_y,N_y-1,1);

    dudy_fft = copy_periodic_boundaries(dudy_fft);
end