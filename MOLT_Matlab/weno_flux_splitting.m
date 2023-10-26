function w = weno_flux_splitting(w,flux,dflux)
    
    epsilon = 1e-6;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Lax-Friedrichs splitting
    % f^{±}(u) = 0.5 * (f(u) ± αu)
    % α = maxu |f′(u)|
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    a = 1.1 * max(abs(dflux(w)));

    v_arr = 0.5 * (flux(w) + a * w);
    u_arr = 0.5 * (flux(w) - a * w);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Choose the positive fluxes: v_i = f^{+}(u_i)
    % to obtain the cell boundary values : v_{i+1/2}^{-}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    vmm = v_arr(1);
    vm =  v_arr(2);
    v =   v_arr(3);
    vp =  v_arr(4);
    vpp = v_arr(5);

    beta_0n = (13.0/12.0)*(vmm - 2*vm +   v)^2 + (1.0/4.0)*(vmm - 4*vm + 3*v)^2;
    beta_1n = (13.0/12.0)*(vm  - 2*v  +  vp)^2 + (1.0/4.0)*(vm  - vp)^2;
    beta_2n = (13.0/12.0)*(v   - 2*vp + vpp)^2 + (1.0/4.0)*(vpp - 4*vp + 3*v)^2;

    d_0n = 1.0/10.0;
    d_1n = 6.0/10.0;
    d_2n = 3.0/10.0;

    alpha_0n = d_0n / (epsilon + beta_0n)^2;
    alpha_1n = d_1n / (epsilon + beta_1n)^2;
    alpha_2n = d_2n / (epsilon + beta_2n)^2;

    alpha_sumn = alpha_0n + alpha_1n + alpha_2n;

    weight_0n = alpha_0n / alpha_sumn;
    weight_1n = alpha_1n / alpha_sumn;
    weight_2n = alpha_2n / alpha_sumn;

    p0 = 2*vmm - 7*vm + 11*v;
    p1 = -vm  + 5*v  + 2*vp;
    p2 = 2*v   + 5*vp - vpp;

    wn  = (1.0/6.0)*( weight_0n*p0 + weight_1n*p1 + weight_2n*p2 );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Choose the negative fluxes: v_i = f^{-}(u_i)
    % to obtain the cell boundary values : v_{i+1/2}^{+}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    umm = u_arr(2);
    um =  u_arr(3);
    u =   u_arr(4);
    up =  u_arr(5);
    upp = u_arr(6);

    beta_0p = (13.0/12.0)*(umm - 2*um +   u)^2. + (1.0/4.0)*(umm - 4*um + 3*u)^2;
    beta_1p = (13.0/12.0)*(um  - 2*u  +  up)^2. + (1.0/4.0)*(um  - up)^2;
    beta_2p = (13.0/12.0)*(u   - 2*up + upp)^2. + (1.0/4.0)*(upp - 4*up + 3*u)^2;

    d_0p = 3.0/10.0;
    d_1p = 6.0/10.0;
    d_2p = 1.0/10.0;

    alpha_0p = d_0p / (epsilon + beta_0p)^2;    
    alpha_1p = d_1p / (epsilon + beta_1p)^2;    
    alpha_2p = d_2p / (epsilon + beta_2p)^2;

    alpha_sump = alpha_0p + alpha_1p + alpha_2p;

    weight_0p = alpha_0p / alpha_sump;
    weight_1p = alpha_1p / alpha_sump;
    weight_2p = alpha_2p / alpha_sump;

    p0 = -umm + 5*um +   2*u;
    p1 = 2*um + 5*u  -    up;
    p2 = 11*u - 7*up + 2*upp;

    wp  = (1.0/6.0)*( weight_0p*(p0) + weight_1p*(p1) + weight_2p*(p2) );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Numerical flux
    % f_{i+1/2} = f_{i+1/2}^{+} + f_{i+1/2}^{-}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    w = wn + wp;
end