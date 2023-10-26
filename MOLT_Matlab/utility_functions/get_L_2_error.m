function L2 = get_L_2_error(U_numerical, U_exact, delta)
    L2 = sqrt( delta*sum(sum((U_numerical - U_exact).^2) ) );
end