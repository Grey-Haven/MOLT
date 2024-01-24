function u = solve_poisson_FFT(RHS, kx_deriv_2, ky_deriv_2)
    [N_y,N_x] = size(RHS);
    
    u_domain = RHS(1:end-1,1:end-1);
    
    u_fft_x = fft(u_domain,N_x-1,2);
    u_fft_xy = fft(u_fft_x,N_y-1,1);
    
    dudx2_fft_over_kx = zeros(N_y-1,N_x-1);

    for i = 1:N_x-1
        for j = 1:N_y-1
            if ~(i == 1 && j == 1)
                dudx2_fft_over_kx(j,i) = -u_fft_xy(j,i) / (kx_deriv_2(i)^2 + ky_deriv_2(j)^2);
                % dudy2_fft_over_ky(j,i) = -u_fft_y(j,i) / (kx_deriv_2(i)^2 + ky_deriv_2(j)^2);
                % dudx2_fft_alt(j,i) = -kx_deriv_2(i)^2 * u_fft_x(j,i);
                % dudy2_fft_alt(j,i) = -kx_deriv_2(j)^2 * u_fft_y(j,i);
            end
        end
    end

    dudx2_fft_y = ifft(dudx2_fft_over_kx, N_x-1, 2);

    u = zeros(N_y,N_x);
    u(1:end-1,1:end-1) = ifft(dudx2_fft_y, N_y-1, 1);
    u = copy_periodic_boundaries(u);
end