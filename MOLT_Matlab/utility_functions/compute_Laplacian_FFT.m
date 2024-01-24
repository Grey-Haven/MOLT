function laplacian_u = compute_Laplacian_FFT(u,kx_deriv_2,ky_deriv_2)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an x derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%

    [N_y,N_x] = size(u);
    
    u_domain = u(1:end-1,1:end-1);

    u_fft_x = fft(u_domain,N_x-1,2);
    u_fft_y = fft(u_domain,N_y-1,1);

    dudx2_fft = zeros(N_y,N_x);
    dudy2_fft = zeros(N_y,N_x);

    dudx2_fft(1:end-1,1:end-1) = ifft(-kx_deriv_2.^2 .*u_fft_x,N_x-1,2);
    dudy2_fft(1:end-1,1:end-1) = ifft(-ky_deriv_2.^2'.*u_fft_y,N_y-1,1);

    dudx2_fft = copy_periodic_boundaries(dudx2_fft);
    dudy2_fft = copy_periodic_boundaries(dudy2_fft);

    laplacian_u = dudx2_fft + dudy2_fft;
end