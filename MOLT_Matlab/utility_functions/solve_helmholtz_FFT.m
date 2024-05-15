function u = solve_helmholtz_FFT(RHS, alpha, kx_deriv_2, ky_deriv_2)

    [N_y,N_x] = size(RHS);
    
    u_domain = RHS(1:end-1,1:end-1);
    
    u_fft_x = fft(u_domain,N_x-1,2);
    u_fft_xy = fft(u_fft_x,N_y-1,1);
    
    invert_L_FFT = zeros(N_y-1,N_x-1);

    for i = 1:N_x-1
        for j = 1:N_y-1
            invert_L_FFT(j,i) = u_fft_xy(j,i) / (1 + (1/(alpha^2))*(kx_deriv_2(i)^2 + ky_deriv_2(j)^2));
        end
    end

    u_inner_y = ifft(invert_L_FFT, N_x-1, 2);

    u = zeros(N_y,N_x);
    u(1:end-1,1:end-1) = ifft(u_inner_y, N_y-1, 1);
    u = copy_periodic_boundaries(u);
    
    imaginary_component_u = imag(u);

    imag_u_inf_norm = max(max(abs(imaginary_component_u)));

    if (imag_u_inf_norm < 1e-12)
        u = real(u);
    else
        ME = MException('HelmholtzSolverException','Imaginary component above machine precision');
        throw(ME);
    end
end