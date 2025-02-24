function inverse = get_L_x_inverse_per_1D(operand, x, dx, dt, c, beta)

    N_x = length(x);

    alpha = beta/(c*dt);
    mu_x = exp(-alpha*( x(end) - x(1) ) );

    % Create arrays for right and left-moving operators
    rite_moving_op = zeros(N_x, 1);
    left_moving_op = zeros(N_x, 1);

    % Extend the data for the integrand along x
    % Corners are not needed
    operand_ext = zeros(N_x, 1+4);

    % Transfer the mesh data
    for i = 1:N_x
        operand_ext(i+2) = operand(i);
    end
    operand_ext = periodic_extension(operand_ext);

    % Invert the x operator and apply to the operand

    % Get the local integrals
    % Note that function names are reversed, as we use the old convention
    Nr = length(rite_moving_op(:));
    Nl = length(left_moving_op(:));
    rite_moving_op(:) = linear5_L(Nr, operand_ext, alpha, dx);
    left_moving_op(:) = linear5_R(Nl, operand_ext, alpha, dx);

    % FC step
    [rite_moving_op(:), left_moving_op(:)] = fast_convolution(rite_moving_op(:), left_moving_op(:), alpha, dx);

    % Combine the integrals into the right-moving operator
    % This gives the convolution integral
    rite_moving_op(:) = .5*(rite_moving_op(:) + left_moving_op(:));

    I_a = rite_moving_op(  1);
    I_b = rite_moving_op(end);

    A_x = I_b/(1-mu_x);
    B_x = I_a/(1-mu_x);

    % Sweep the x boundary data into the operator
    inverse = apply_A_and_B(rite_moving_op(:), x, alpha, A_x, B_x);
end