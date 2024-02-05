function u = richardson_extrapolation(u_0, RHS, kappa, dt, kx_deriv_2, ky_deriv_2)

    beta_BDF = 1;

    alpha_1 = beta_BDF/(kappa*dt);
    alpha_2 = beta_BDF/(kappa*(dt/2));

    S_1 = u_0 + 1/alpha_1^2 * RHS;
    S_2 = u_0 + 1/alpha_2^2 * RHS;

    u_1 = solve_helmholtz_FFT(S_1, alpha_1, kx_deriv_2, ky_deriv_2);
    u_2 = solve_helmholtz_FFT(S_2, alpha_2, kx_deriv_2, ky_deriv_2);

    u = 4/3*u_2 - 1/3*u_1;

end