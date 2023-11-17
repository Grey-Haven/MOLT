function r = rho_implicit(rho_guess,rho_n,u,dt,kx,ky)

    u1 = u(:,:,1);
    u2 = u(:,:,2);

    Nx = size(rho_guess,1);
    Ny = size(rho_guess,2);

    J1 = rho_guess.*u1;
    J2 = rho_guess.*u2;

    J1_fft_deriv_x = ifft(sqrt(-1)*kx'.*fft(J1,Nx,1),Nx,1);
    J2_fft_deriv_y = ifft(sqrt(-1)*ky .*fft(J2,Ny,2),Ny,2);

    div_J = J1_fft_deriv_x + J2_fft_deriv_y;

    r = rho_guess - rho_n + dt*div_J;

end