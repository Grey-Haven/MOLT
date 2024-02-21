function v_ext = polynomial_extension(v)
    %%%
    % Fills the ghost region of an array "v" using polynomial extrapolation.
    %
    % Assumes that v is a 1-D array which has 2 ghost points on each end.
    %
    % Note: v includes the extension, so indices are adjusted accordingly.
    %%%

%     v[0] = 15*v[2]- 40*v[3] + 45*v[4] -24*v[5] + 5*v[6]
%     v[1] =  5*v[2]- 10*v[3] + 10*v[4] - 5*v[5] +   v[6]

    v_ext = v;

    % Left region
    v_ext(1) = 15*v(3) - 40*v(4) + 45*v(5) - 24*v(6) + 5*v(7);
    v_ext(2) =  5*v(3) - 10*v(4) + 10*v(5) -  5*v(6) +   v(7);

%     v[-2] =  5*v[-3] - 10*v[-4] + 10*v[-5] -  5*v[-6] +   v[-7]
%     v[-1] = 15*v[-3] - 40*v[-4] + 45*v[-5] - 24*v[-6] + 5*v[-7]

    % Right region    
    v_ext(end-1) =  5*v(end-2) - 10*v(end-3) + 10*v(end-4) -  5*v(end-5) +   v(end-6);
    v_ext(end  ) = 15*v(end-2) - 40*v(end-3) + 45*v(end-4) - 24*v(end-5) + 5*v(end-6);

end