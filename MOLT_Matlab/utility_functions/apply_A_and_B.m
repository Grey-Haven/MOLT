function [] = apply_A_and_B(I, x, alpha, A, B)
    
    N = length(x);
    
    for i = 1:N
        I(i) = I(i) + A*exp(-alpha*( x(i  ) - x(1) ));
        I(i) = I(i) + B*exp(-alpha*( x(end) - x(i) ));
    end
end